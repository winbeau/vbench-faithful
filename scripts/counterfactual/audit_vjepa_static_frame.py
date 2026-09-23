"""Verify the user-requested still/pan controls and report, without calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .audit_vjepa_motion_probe import rows, sha
from .official_video_jitter import native_video
from .vjepa_motion_probe import fresh_output, write_json


def audit(root, previous_root):
    root, previous_root = Path(root), Path(previous_root)
    receipt = json.loads((root / 'construction/completion.json').read_text())
    inputs = rows(root / 'construction/inputs.jsonl')
    assert sha(root / 'construction/inputs.jsonl') == receipt['manifest_sha256']
    assert sha(root / 'construction/source_frame.png') == receipt['source_frame_sha256']
    frame = cv2.cvtColor(cv2.imread(str(root / 'construction/source_frame.png')), cv2.COLOR_BGR2RGB)
    output = {}
    for backend in ('origin', 'vjepa'):
        directory = root / 'scores' / backend
        completion = json.loads((directory / 'completion.json').read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        driver = json.loads((directory / 'diagnostic_driver.json').read_text())
        assert completion['status'] == 'finished' and completion['failed'] == 0
        assert completion['completed'] == completion['expected'] == 5
        assert sha(directory / 'scores.jsonl') == completion['scores_sha256']
        assert provenance['input_sha256'] == receipt['manifest_sha256']
        assert provenance['config_sha256'] == receipt['config_sha256']
        assert sha(root / 'code/scripts/counterfactual/probe_vjepa_static_frame.py') == driver['driver_sha256'] == receipt['driver_sha256']
        for path, checksum in provenance['code'].items():
            assert sha(root / 'code' / path) == checksum
        output[backend] = {r['family']: r for r in rows(directory / 'scores.jsonl')}
        assert set(output[backend]) == {'still', 'pan8', 'pan32', 'local_jitter8', 'original'}
    table = []
    for row in inputs:
        family = row['family']
        o, v = output['origin'][family], output['vjepa'][family]
        assert o['status'] == v['status'] == 'ok' and o['input_sha256'] == v['input_sha256'] == row['sha256']
        if family != 'original':
            video = root / 'construction/videos' / f'{family}.mp4'
            assert sha(video) == row['sha256']
            decoded, pts, fps = native_video(video, 1e-6)
            assert decoded.shape == (16, *frame.shape) and fps == 8
            assert np.allclose(pts, np.arange(16) / 8, atol=1e-6, rtol=0)
            if family == 'still':
                assert np.array_equal(decoded, np.repeat(frame[None], 16, axis=0))
            if family.startswith('pan'):
                distance = int(family[3:])
                assert np.array_equal(decoded[0], frame)
                assert np.array_equal(decoded[-1, :, distance:], frame[:, :-distance])
            difference = float(np.abs(np.diff(decoded.astype(float), axis=0)).mean())
            assert abs(difference - row['mean_interframe_absolute_rgb_difference']) < 1e-12
        for arm in ('joint', 'natural_only'):
            assert abs(1 / (1 + np.exp(-v[arm]['latent'])) - v[arm]['score']) < 1e-12
        diagnostics = o['diagnostics']; raw = diagnostics['official']; sample = diagnostics['sampling']
        threshold = 6 * min(sample['frame_shape']) / 256
        expected = float(np.count_nonzero(np.array(raw['raw_flow_top5_mean']) > threshold) >= round(4 * sample['sampled_frame_count'] / 16))
        assert o['score'] == expected
        table.append({'case': family, 'origin': o['score'], 'joint': v['joint'], 'natural_only': v['natural_only'],
                      'mean_interframe_absolute_rgb_difference': row['mean_interframe_absolute_rgb_difference']})
    prior = next(r for p in (previous_root / 'scores/vjepa').glob('shard-*/scores.jsonl') for r in rows(p)
                 if r['base_id'] == receipt['source_uid'] and r['family'] == 'original')
    assert prior['input_sha256'] == output['vjepa']['original']['input_sha256']
    assert prior['feature_sha256'] == output['vjepa']['original']['feature_sha256']
    assert all(prior[arm] == output['vjepa']['original'][arm] for arm in ('joint', 'natural_only'))
    scores = {r['case']: r['joint']['score'] for r in table}
    return {'integrity_checks': 'passed', 'source_uid': receipt['source_uid'], 'inputs_per_backend': 5,
            'native_shape': receipt['native_shape'], 'fps': 8, 'duration_seconds': 2, 'rows': table,
            'findings': {'still_score': scores['still'], 'pan8_minus_still': scores['pan8'] - scores['still'],
                         'pan32_minus_still': scores['pan32'] - scores['still'],
                         'jitter8_minus_still': scores['local_jitter8'] - scores['still'],
                         'original_minus_still': scores['original'] - scores['still'],
                         'monotonic_still_pan8_pan32': scores['still'] <= scores['pan8'] <= scores['pan32']},
            'score_mapping_changed': False, 'original_feature_and_scores_exact': True,
            'scope': receipt['scope'], 'calibration_fit': False, 'auditor_sha256': sha(__file__)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous-root', 'output'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    result = audit(args.root, args.previous_root)
    out = fresh_output(args.output)
    write_json(out / 'summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
