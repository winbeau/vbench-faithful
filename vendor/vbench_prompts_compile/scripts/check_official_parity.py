#!/usr/bin/env python3
"""Compare CPU replay against AST-extracted functions from the locked upstream.

No import of the official GPU/model stack and no modification of its checkout.
Run on the scoring host with --vbench-root; archive the report with its file hashes.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vbench_prompts_compile import annotation_jobs as J
from vbench_prompts_compile import official_replay as O


def extract(path, names):
    module = ast.parse(path.read_text())
    nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('upstream function missing')
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def verify(directory, trials=2000):
    sp = extract(directory / 'spatial_relationship.py', ['get_position_score', 'check_generate'])
    sc = extract(directory / 'scene.py', ['check_generate'])
    ob = extract(directory / 'multiple_objects.py', ['check_generate'])
    rng = random.Random(20260919)
    relations = ['on the left of', 'on the right of', 'on the top of', 'on the bottom of']
    def box():
        x, y = rng.uniform(0, 600), rng.uniform(0, 400)
        return [x, y, x + rng.uniform(1, 180), y + rng.uniform(1, 180)]
    for _ in range(trials):
        a, b, relation = box(), box(), rng.choice(relations)
        assert abs(sp['get_position_score'](relation, a, b) - O.position_score(relation, a, b)) < 1e-12
        frames = [[{'label': rng.choice(['cat', 'dog', 'other']), 'box': box()} for _ in range(rng.randrange(8))] for _ in range(3)]
        target = {'object_a': 'cat', 'object_b': 'dog', 'relationship': relation}
        native = sp['check_generate'](target, [[[item['label'], item['box']] for item in frame] for frame in frames])
        assert all(abs(a - b) < 1e-12 for a, b in zip(native, O.spatial_scores(target, frames)))
    captions = ['an ocean', 'a sea', 'Ocean waves', 'a seaside street', 'a forest', '']
    scene_cases = ['ocean', 'sea', 'a street', 'Ocean', ' ocean', '']
    for key in scene_cases:
        assert sc['check_generate']({'scene': key}, captions) == sum(O.scene_scores(key, captions))
    labels = [{'cat', 'dog'}, {'cat'}, {'Cat', 'dog'}, set()]
    assert ob['check_generate']('cat and dog', labels) == sum(O.object_scores('cat and dog', labels))
    # Counterexamples establish why the prior replay was not official parity.
    target = {'object_a': 'cat', 'object_b': 'dog', 'relationship': 'on the left of'}
    two_cats = [[{'label': 'cat', 'box': [0, 0, 10, 10]}, {'label': 'cat', 'box': [30, 0, 40, 10]}]]
    assert O.spatial_scores(target, two_cats) == [1.0]
    assert O.position_score('on the left of', [0, 0, 10, 10], [30, 0, 40, 10]) == O.position_score('on the right of', [0, 0, 10, 10], [30, 0, 40, 10])
    return {'passed': True, 'random_geometry_cases': trials, 'random_frame_cases': 3 * trials,
            'scene_cases': len(scene_cases), 'objects_cases': len(labels),
            'findings': ['spatial direction sign is ignored upstream', 'two detections of object_a can satisfy object_a/object_b upstream'],
            'scope': 'pure decision-rule parity; no claim of GPU/model parity',
            'sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ['spatial_relationship.py', 'scene.py', 'multiple_objects.py']}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vbench-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    root = Path(args.vbench_root)
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != O.UPSTREAM_SHA:
        raise ValueError('upstream revision differs from frozen protocol')
    for name in ['spatial_relationship.py', 'scene.py', 'multiple_objects.py']:
        committed = subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:vbench/{name}'])
        if committed != (root / 'vbench' / name).read_bytes():
            raise ValueError(f'upstream working file differs from frozen commit: {name}')
    report = {**verify(root / 'vbench'), 'upstream_commit': sha}
    J.atomic_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
