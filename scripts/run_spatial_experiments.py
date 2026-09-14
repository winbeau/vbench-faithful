#!/usr/bin/env python3
"""Resume-safe Spatial Relationship experiment dispatcher.

This is a thin experiment adapter. It does not change GRiT, frame sampling,
relation geometry, thresholds, role assignment, or temporal aggregation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY = Path(os.environ.get('SPATIAL_GRIT_OVERLAY', '/root/autodl-tmp/vbench-audit-storage/envs/spatial-grit/site-packages'))
for path in (OVERLAY, REPO_ROOT / '指标/spatial_relationship/src', REPO_ROOT / '公共/audit-core/src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VARIANTS = ('official', 'signed_direction', 'role_preserving', 'full_temporal')
FAMILIES = ('directional_inversion', 'role_swap', 'temporal_persistence', 'multi_instance', 'detection_control')
REQUIRED = ('base_id', 'derived_id', 'intervention_family', 'intervention_level', 'subject_a', 'relation', 'subject_b', 'expected_relation')
RELATIONS = {
    'left': 'on the left of', 'on the left of': 'on the left of',
    'right': 'on the right of', 'on the right of': 'on the right of',
    'top': 'on the top of', 'on the top of': 'on the top of',
    'bottom': 'on the bottom of', 'on the bottom of': 'on the bottom of',
}


class VariantNotReady(RuntimeError):
    pass


def variant_mode(variant: str) -> str | None:
    mapping = {'official': None, 'signed_direction': 'signed_only', 'role_preserving': 'ordered_role_identity_assignment'}
    if variant == 'full_temporal':
        raise VariantNotReady('full_temporal is not implemented: the current evaluator has no temporal persistence component')
    if variant not in mapping:
        raise ValueError(f'unknown variant: {variant}')
    return mapping[variant]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'malformed JSONL {path}:{line_number}: {exc}') from exc
        if not isinstance(row, dict):
            raise ValueError(f'manifest row must be an object: {path}:{line_number}')
        rows.append(row)
    return rows


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Object of type {value.__class__.__name__} is not JSON serializable')


def manifest_key(row: Mapping[str, Any], variant: str) -> tuple[str, ...]:
    return tuple(str(row[name]) for name in ('base_id', 'derived_id', 'intervention_family', 'intervention_level', 'subject_a', 'relation', 'subject_b')) + (variant,)


def result_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[name]) for name in ('base_id', 'derived_id', 'intervention_family', 'intervention_level', 'subject_a', 'relation', 'subject_b', 'metric_variant'))


def pending_rows(rows: list[dict[str, Any]], existing: set[tuple[str, ...]], variant: str) -> list[dict[str, Any]]:
    return [row for row in rows if manifest_key(row, variant) not in existing]


def validate_manifest(rows: list[dict[str, Any]], manifest: Path, family: str, data_root: Path | None) -> list[dict[str, Any]]:
    selected = []
    keys = set()
    for line_number, row in enumerate(rows, 1):
        missing = [name for name in REQUIRED if name not in row or row[name] in (None, '')]
        if missing:
            raise ValueError(f'manifest row {line_number} missing fields: {missing}')
        if row['intervention_family'] not in FAMILIES:
            raise ValueError(f'manifest row {line_number} has unsupported family: {row["intervention_family"]}')
        if row['intervention_family'] != family:
            continue
        raw_path = row.get('video_path', row.get('path'))
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f'manifest row {line_number} needs explicit video_path or path')
        video = Path(raw_path).expanduser()
        if not video.is_absolute():
            video = (data_root if data_root is not None else manifest.parent) / video
        video = video.resolve()
        if not video.is_file():
            raise FileNotFoundError(f'manifest video is missing: {video}')
        normalized = dict(row)
        normalized['video_path'] = str(video)
        relation = str(normalized['relation']).strip().lower()
        if relation not in RELATIONS:
            raise ValueError(f'manifest row {line_number} has unsupported relation: {normalized["relation"]}')
        normalized['relation'] = RELATIONS[relation]
        key = manifest_key(normalized, '__manifest__')
        if key in keys:
            raise ValueError(f'duplicate manifest evaluation key at row {line_number}: {key}')
        keys.add(key)
        selected.append(normalized)
    if not selected:
        raise ValueError(f'no manifest rows selected for family={family}')
    return selected


def detection_diagnostics(diagnostics: Any, subject: str, object_: str) -> dict[str, Any]:
    if not isinstance(diagnostics, Mapping):
        return {'detected_a': None, 'detected_b': None, 'detection_confidence': None, 'selected_instance_pair': None}
    frames = diagnostics.get('frame_results')
    if isinstance(frames, list):
        detected_a = any(bool(frame.get('subject_candidate_ids')) for frame in frames)
        detected_b = any(bool(frame.get('object_candidate_ids')) for frame in frames)
        confidences = []
        selected = []
        for frame in frames:
            detections = {item.get('detection_id'): item for item in frame.get('detections', [])}
            subject_id, object_id = frame.get('assigned_subject_id'), frame.get('assigned_object_id')
            subject_det, object_det = detections.get(subject_id), detections.get(object_id)
            if subject_det is not None or object_det is not None:
                selected.append({'frame_index': frame.get('frame_index'), 'subject_id': subject_id, 'object_id': object_id, 'subject_box': frame.get('selected_subject_box'), 'object_box': frame.get('selected_object_box')})
            confidences.append({'frame_index': frame.get('frame_index'), 'subject': None if subject_det is None else subject_det.get('confidence'), 'object': None if object_det is None else object_det.get('confidence')})
        return {'detected_a': detected_a, 'detected_b': detected_b, 'detection_confidence': confidences, 'selected_instance_pair': selected}
    frame_scores = diagnostics.get('frame_scores')
    if isinstance(frame_scores, list):
        total = len(frame_scores)
        return {
            'detected_a': None if total == 0 else diagnostics.get('missing_subject_count', total) < total,
            'detected_b': None if total == 0 else diagnostics.get('missing_object_count', total) < total,
            'detection_confidence': None,
            'selected_instance_pair': None,
        }
    return {'detected_a': None, 'detected_b': None, 'detection_confidence': None, 'selected_instance_pair': None}


def _base_record(row: Mapping[str, Any], variant: str) -> dict[str, Any]:
    return {
        'base_id': str(row['base_id']), 'derived_id': str(row['derived_id']),
        'intervention_family': str(row['intervention_family']), 'intervention_level': row['intervention_level'],
        'subject_a': str(row['subject_a']), 'relation': str(row['relation']), 'subject_b': str(row['subject_b']),
        'expected_relation': row['expected_relation'], 'metric_variant': variant, 'video_path': str(row['video_path']),
    }


def evaluate_row(row: Mapping[str, Any], variant: str, evaluator: Any, diagnostics_level: Any) -> dict[str, Any]:
    from spatial_relationship.backends.audit import score_predictions
    from spatial_relationship.models import AblationMode
    from spatial_relationship.relation import normalize_relation
    from spatial_relationship.schemas import OrderedRelationQuery

    started = time.monotonic()
    record = _base_record(row, variant)
    query = OrderedRelationQuery(record['subject_a'], normalize_relation(record['relation']), record['subject_b'])
    record['relation'] = query.relation
    try:
        if variant == 'official':
            metadata = {'prompt': str(row.get('prompt', '')), 'dimension_metadata': {'object_a': query.subject, 'object_b': query.object, 'relationship': query.relation}}
            raw = evaluator.evaluate_video(Path(record['video_path']), metadata, query)
            _, items = raw
            if len(items) != 1:
                raise RuntimeError(f'official single-video result count is {len(items)}, expected 1')
            score = float(items[0]['video_results'])
            diagnostics = {'frame_results': items[0].get('frame_results'), 'video_results': score, 'official_entrypoint': 'vbench.spatial_relationship.spatial_relationship'}
            detection = detection_diagnostics(None, query.subject, query.object)
            provenance = {'mode': 'locked_upstream_official', 'temporal_aggregation': 'mean_over_16_sampled_frames'}
        else:
            frame_indices, predictions = evaluator.detect_video(Path(record['video_path']))
            mode = AblationMode(variant_mode(variant))
            scored = score_predictions(record['video_path'], str(row.get('prompt', '')), query, predictions, sampled_frame_indices=frame_indices, mode=mode)
            diagnostics = scored.to_dict(include_frames=diagnostics_level.value == 'full')
            score = float(scored.video_score)
            detection = detection_diagnostics(diagnostics, query.subject, query.object)
            provenance = {'mode': mode.value, 'temporal_aggregation': scored.aggregation_method, 'detector': 'locked_official_grit_objectdet'}
        return {**record, 'score': score, **detection, 'relation_score': score, 'failure_or_abstention': None, 'component_provenance': provenance, 'diagnostics': diagnostics, 'runtime_s': time.monotonic() - started}
    except Exception as exc:
        return {**record, 'score': None, 'detected_a': None, 'detected_b': None, 'detection_confidence': None, 'selected_instance_pair': None, 'relation_score': None, 'failure_or_abstention': f'{type(exc).__name__}: {exc}', 'component_provenance': None, 'diagnostics': None, 'runtime_s': time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description='Resume-safe Spatial Relationship experiment runner')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--weight', type=Path, default=Path(os.environ.get('VBENCH_AUDIT_GRIT_WEIGHT', '/root/autodl-tmp/vbench-audit-storage/models/grit/grit_b_densecap_objectdet.pth')))
    parser.add_argument('--upstream', type=Path, default=Path(os.environ.get('VBENCH1_ROOT', '/root/vbench1')))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--diagnostics', choices=('full', 'compact'), default='compact')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()

    variant_mode(args.variant)
    rows = validate_manifest(read_jsonl(args.manifest), args.manifest, args.family, args.data_root)
    if args.validate_only:
        print(json.dumps({'status': 'VALID', 'rows': len(rows), 'family': args.family, 'variant': args.variant}))
        return 0
    if not args.weight.is_file():
        raise FileNotFoundError(f'GRiT weight not found: {args.weight}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if args.output.exists():
        for row in read_jsonl(args.output):
            existing.add(result_key(row))
    pending = pending_rows(rows, existing, args.variant)
    skipped = len(rows) - len(pending)
    if not pending:
        print(json.dumps({'status': 'COMPLETE', 'variant': args.variant, 'family': args.family, 'selected': len(rows), 'evaluated': 0, 'skipped': skipped, 'succeeded': 0, 'failed': 0, 'output': str(args.output)}, ensure_ascii=False))
        return 0

    from spatial_relationship.backends.vbench import OfficialGritDetector, OfficialVBenchEvaluator, verify_upstream
    from spatial_relationship.diagnostics import DiagnosticsLevel
    import torch

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    verify_upstream(args.upstream)
    device = torch.device(args.device)
    evaluator = OfficialVBenchEvaluator(device, args.weight, args.upstream) if args.variant == 'official' else OfficialGritDetector(device, args.weight, args.upstream)
    level = DiagnosticsLevel(args.diagnostics)
    succeeded = failed = 0
    with args.output.open('a', encoding='utf-8') as handle:
        for row in pending:
            result = evaluate_row(row, args.variant, evaluator, level)
            try:
                serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, default=json_default)
            except Exception as exc:
                result = {**_base_record(row, args.variant), 'score': None, 'detected_a': None, 'detected_b': None, 'detection_confidence': None, 'selected_instance_pair': None, 'relation_score': None, 'failure_or_abstention': f'structured evidence serialization failed: {type(exc).__name__}: {exc}', 'component_provenance': None, 'diagnostics': None, 'runtime_s': result.get('runtime_s')}
                serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
            handle.write(serialized + '\n')
            handle.flush()
            if result['failure_or_abstention'] is None:
                succeeded += 1
            else:
                failed += 1
    print(json.dumps({'status': 'COMPLETE', 'variant': args.variant, 'family': args.family, 'selected': len(rows), 'evaluated': len(pending), 'skipped': skipped, 'succeeded': succeeded, 'failed': failed, 'output': str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
