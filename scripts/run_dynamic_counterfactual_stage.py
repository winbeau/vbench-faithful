#!/usr/bin/env python3
"""Resume-safe formal Dynamic Degree stage runner (one split and variant)."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from dynamic_degree.backends.audit import AuditConfig, AuditVariant, analyze_timed_flow_sequence, audit_result_payload
from dynamic_degree.backends.vbench import OfficialDynamicEvaluator, official_result_payload
from dynamic_degree.diagnostics import DiagnosticsLevel
from dynamic_degree.evaluation_records import EvaluationRecordWriter, record_key
from dynamic_degree.formal_results import normalize_batch_result
from dynamic_degree.models import RaftFlowModel, decode_timed_frames
from dynamic_degree.prompt_target import parse_motion_target
from vbench_audit_core.upstream import resolve_upstream_path

FAMILIES = ('fps_resampling', 'resolution', 'subject_speed', 'motion_coverage', 'camera_shake', 'subject_camera', 'static_flicker')
VARIANTS = ('official', 'time_only', 'source_only', 'duration_only', 'source_time', 'full')
AUDIT_VARIANTS = {name: AuditVariant(name) for name in VARIANTS if name != 'official'}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'malformed JSONL {path}:{number}: {exc}') from exc
    return rows


class SidecarWriter:
    KEY_FIELDS = ('base_id', 'derived_id', 'split', 'metric_variant', 'target_type')

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = set()
        if path.exists():
            for number, row in enumerate(read_jsonl(path), 1):
                if not all(field in row for field in self.KEY_FIELDS) or 'result' not in row:
                    raise RuntimeError(f'malformed sidecar record {path}:{number}')
                key = tuple(str(row[field]) for field in self.KEY_FIELDS)
                if key in self.keys:
                    raise RuntimeError(f'duplicate sidecar key {key}')
                self.keys.add(key)

    def append(self, row: dict[str, Any]) -> bool:
        key = tuple(str(row[field]) for field in self.KEY_FIELDS)
        if key in self.keys:
            return False
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            handle.flush()
        self.keys.add(key)
        return True


def target_type(variant: str, prompt: str) -> str:
    if variant != 'full':
        return 'GENERIC'
    return parse_motion_target(prompt).target.upper()


def run_one(video: Path, prompt: str, target: str, variant: str, backend: Any, config: AuditConfig | None) -> dict[str, Any]:
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(0)
    try:
        if variant == 'official':
            value = backend.evaluate_video(video)
            result = {
                'video': str(video), 'prompt': prompt, 'backend': 'vbench',
                'score': float(value.official_video_boolean), 'status': 'succeeded',
                'failure_reason': None, 'structured_evidence': {'official_video_boolean': value.official_video_boolean},
                'diagnostics': official_result_payload(value),
            }
        else:
            sequence = decode_timed_frames(video)
            audit = analyze_timed_flow_sequence(str(video), prompt, sequence, backend, motion_target_override=target.lower(), config=config)
            result = audit_result_payload(audit, DiagnosticsLevel.COMPACT)
            result['structured_evidence'] = {
                'apparent_intensity': result.get('apparent_intensity'), 'camera_intensity': result.get('camera_intensity'),
                'residual_intensity': result.get('residual_intensity'), 'apparent_coverage': result.get('apparent_coverage'),
                'camera_coverage': result.get('camera_coverage'), 'residual_coverage': result.get('residual_coverage'),
                'task_relevant_motion_evidence': result.get('task_relevant_motion_evidence'),
                'component_provenance': (result.get('diagnostics') or {}).get('audit_video', {}).get('component_provenance'),
                'routing': {'target_type': target, 'selected_evidence_channel': result.get('selected_evidence_channel')},
            }
    except Exception as exc:
        result = {'video': str(video), 'prompt': prompt, 'backend': variant, 'score': None, 'status': 'failed',
                  'failure_reason': f'{type(exc).__name__}: {exc}', 'structured_evidence': None, 'diagnostics': None}
    result = normalize_batch_result(result)
    result['runtime_s'] = time.monotonic() - started
    result['peak_gpu_memory_mb'] = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--evaluations', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--stage-status', type=Path, required=True)
    parser.add_argument('--weight', type=Path, required=True)
    args = parser.parse_args()

    meta = args.dataset / 'metadata'
    if not (meta / 'DATASET_FROZEN').is_file():
        raise RuntimeError('DATASET_FROZEN missing')
    bases = {row['base_id']: row for row in read_jsonl(meta / 'base_manifest.jsonl')}
    derived = read_jsonl(meta / 'derived_manifest.jsonl')
    if len(bases) != 50 or len(derived) != 730:
        raise RuntimeError('frozen manifest cardinality mismatch')
    selected = [row for row in derived if row.get('split') == args.split]
    if any(row['base_id'] not in bases or bases[row['base_id']].get('split') != args.split for row in selected):
        raise RuntimeError('manifest split leakage')
    if {row['intervention_family'] for row in selected} != set(FAMILIES):
        raise RuntimeError('family coverage mismatch')
    selected.sort(key=lambda row: (FAMILIES.index(row['intervention_family']), row['base_id'], str(row['intervention_level']), row['derived_id']))

    writer = EvaluationRecordWriter(args.evaluations)
    sidecar = SidecarWriter(args.evidence)
    existing = writer._keys.copy()
    pending = []
    for row in selected:
        prompt = str(bases[row['base_id']]['prompt'])
        target = target_type(args.variant, prompt)
        probe = {'base_id': row['base_id'], 'derived_id': row['derived_id'], 'metric_name': 'dynamic_degree', 'metric_variant': args.variant, 'target_type': target}
        if record_key(probe) not in existing:
            pending.append((row, prompt, target))

    backend = config = None
    if pending:
        device = torch.device('cuda:0')
        if args.variant == 'official':
            backend, config = OfficialDynamicEvaluator(device, args.weight), None
        else:
            backend, config = RaftFlowModel(device, args.weight, resolve_upstream_path()), AuditConfig(variant=AUDIT_VARIANTS[args.variant])
    else:
        print(json.dumps({'event': 'existing_records_skipped', 'split': args.split, 'variant': args.variant, 'count': len(selected)}), flush=True)

    counters = Counter()
    for family in FAMILIES:
        family_rows = [item for item in pending if item[0]['intervention_family'] == family]
        print(json.dumps({'event': 'family_start', 'split': args.split, 'variant': args.variant, 'family': family, 'pending': len(family_rows)}), flush=True)
        for row, prompt, target in family_rows:
            video = args.dataset / row['path']
            result = run_one(video, prompt, target, args.variant, backend, config)
            join = {'base_id': row['base_id'], 'derived_id': row['derived_id'], 'split': args.split, 'metric_variant': args.variant, 'target_type': target}
            sidecar.append({**join, 'intervention_family': family, 'result': result})
            record = {**join, 'dimension': 'dynamic_degree', 'intervention_family': family,
                'intervention_level': row['intervention_level'], 'prompt': prompt, 'expected_relation': row['expected_relation'],
                'metric_name': 'dynamic_degree', 'score': result['score'], 'seed': row.get('seed'),
                'failure_or_abstention': result['failure_reason']}
            if not writer.append(record):
                raise RuntimeError(f'unexpected duplicate while appending: {record_key(record)}')
            counters[result['status']] += 1
        print(json.dumps({'event': 'family_complete', 'split': args.split, 'variant': args.variant, 'family': family}), flush=True)

    all_records = read_jsonl(args.evaluations)
    stage_records = [row for row in all_records if row['split'] == args.split and row['metric_variant'] == args.variant]
    keys = [record_key(row) for row in stage_records]
    expected_ids = {row['derived_id'] for row in selected}
    actual_ids = {row['derived_id'] for row in stage_records}
    sidecar_stage = [row for row in read_jsonl(args.evidence) if row['split'] == args.split and row['metric_variant'] == args.variant]
    status = {'split': args.split, 'variant': args.variant, 'expected_records': len(selected), 'actual_records': len(stage_records),
        'success': sum(row['failure_or_abstention'] is None for row in stage_records),
        'failure_or_abstention': sum(row['failure_or_abstention'] is not None for row in stage_records),
        'duplicates': len(keys) - len(set(keys)), 'missing': sorted(expected_ids - actual_ids), 'unexpected': sorted(actual_ids - expected_ids),
        'sidecar_records': len(sidecar_stage), 'new_status_counts': dict(counters)}
    args.stage_status.parent.mkdir(parents=True, exist_ok=True)
    args.stage_status.write_text(json.dumps(status, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    integrity_ok = status['actual_records'] == status['expected_records'] and not status['duplicates'] and not status['missing'] and not status['unexpected'] and status['sidecar_records'] == status['expected_records']
    print(json.dumps({'event': 'stage_complete', **status, 'integrity_ok': integrity_ok}), flush=True)
    return 0 if integrity_ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
