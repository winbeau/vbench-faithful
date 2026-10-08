"""Frozen manifest and scene-condition helpers used by paper data builders.

Extracted from the published semantic construction snapshot; scoring and model
initialization stay in their existing runtime modules.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile.experiments import SCENE_SYNONYMS, bare_scene
from scripts.semantic import audit_scene_sources as audit

DIMENSIONS = {'spatial': 'spatial_relationship', 'objects': 'multiplt_object', 'action': 'human_action', 'scene': 'scene'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows_from_manifest(path, limit=None, per_prompt=None):
    by_prompt = {}
    with Path(path).open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            prompt = (row.get('prompt_en') or '').strip()
            if not prompt:
                continue
            videos = by_prompt.setdefault(prompt, [])
            for key in ['video_a_path', 'video_b_path']:
                relative = row.get(key)
                if relative and relative not in videos:
                    videos.append(relative)
    cap = max((len(v) for v in by_prompt.values()), default=0)
    if per_prompt is not None:
        cap = min(cap, per_prompt)
    selected = []
    for i in range(cap):
        for prompt, videos in by_prompt.items():
            if i < len(videos):
                selected.append({'prompt': prompt, 'relative_path': videos[i]})
    return selected[:limit] if limit is not None else selected


def normalize_record(row):
    return {**row, 'input': row.get('input', {'prompt': row.get('prompt'), 'caption': row.get('caption')}),
        'meta': {**row.get('meta', {}), 'image': row.get('meta', {}).get('image', row.get('image'))}}


def build_conditions(gold_rows, synonym_rows=None, *, plan_rows=None):
    gold_rows = [normalize_record(r) for r in gold_rows]
    planned = [normalize_record(r) for r in (plan_rows if plan_rows is not None else gold_rows)]
    by_id = defaultdict(list)
    for row in gold_rows:
        by_id[J.observation_id(row)].append(row)
    components = audit.components(planned)
    conditions = []
    for row, family in zip(planned, components):
        labels = by_id.get(J.observation_id(row), [])
        expected = {r.get('meta', {}).get('caption_evidence_label') for r in labels} - {None}
        visual = {r.get('meta', {}).get('visual_truth') for r in labels} - {None}
        conditions.append({'condition': row.get('meta', {}).get('pair_type', 'unknown'),
            'expected': next(iter(expected)) if len(expected) == 1 else None,
            'visual_expected': next(iter(visual)) if len(visual) == 1 else None,
            'prompt': row['input']['prompt'], 'caption': row['input']['caption'],
            'family': family, 'source_id': J.observation_id(row),
            'image': row['meta'].get('image'), 'video_path': row['meta'].get('frame_video'),
            'frame_index': row['meta'].get('frame_index')})
    originals = list(conditions)
    unmatched = 0
    if synonym_rows is None:
        for row in originals:
            key = bare_scene(row['prompt'])
            if key in SCENE_SYNONYMS:
                synonym = SCENE_SYNONYMS[key]
                article = 'an ' if synonym[0].lower() in 'aeiou' else 'a '
                conditions.append({**row, 'condition': 'synonym', 'original_prompt': row['prompt'],
                    'original_expected': row['expected'], 'prompt': article + synonym,
                    'source_id': row['source_id'] + ':synonym'})
    else:
        for synonym in synonym_rows:
            matched = [r for r in originals if r['prompt'] == synonym['original_prompt'] and r['caption'] == synonym['caption']]
            if not matched:
                unmatched += 1
            for row in matched:
                conditions.append({**row, 'condition': 'synonym', 'original_prompt': row['prompt'],
                    'original_expected': row['expected'], 'prompt': synonym['variant_prompt'],
                    'source_id': row['source_id'] + ':synonym'})
    return conditions, unmatched
