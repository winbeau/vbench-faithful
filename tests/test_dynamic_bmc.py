import json
import zipfile

import numpy as np
import pytest

from scripts.counterfactual.evaluate_dynamic_bmc import group_statistics, safe_zip, sampling_windows


@pytest.mark.parametrize('fps', [8, 15, 25, 29.97, 30])
def test_native_time_sampling_is_distinct_nonoverlapping_and_nearest(fps):
    windows = sampling_windows(3000, fps)
    assert len(windows) == 12
    assert windows[0]['indices'][0] == 0
    for w in windows:
        assert len(w['indices']) == len(set(w['indices'])) == 16
        assert w['max_rounding_seconds'] <= .5 / fps + 1e-12
        assert np.diff(w['target_seconds']) == pytest.approx(np.repeat(.125, 15))
    assert all(a['indices'][-1] < b['indices'][0] for a, b in zip(windows, windows[1:]))


def test_short_recording_reduces_count_without_repeating_frames():
    assert len(sampling_windows(125, 25)) == 2
    with pytest.raises(ValueError):
        sampling_windows(49, 25)
    with pytest.raises(ValueError):
        sampling_windows(100, 4)


def test_archive_cannot_escape_output(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape.txt', 'no')
    with pytest.raises(ValueError, match='unsafe'):
        safe_zip(archive, tmp_path / 'output')
    assert not (tmp_path / 'escape.txt').exists()


def test_analysis_keeps_ambiguous_in_primary_and_counts_cf_not_just_mean():
    config = {'analysis': {'bootstrap_seed': 1, 'bootstrap_replicates': 100}}
    pairs = [{'sequence': 'a', 'condition': 'static', 'origin': [0, 1, 1], 'repair': [.1, .3, .0]},
             {'sequence': 'b', 'condition': 'motion', 'origin': [1, 1, 1], 'repair': [.6, .5, .7]},
             {'sequence': 'c', 'condition': 'ambiguous', 'origin': [0, 0, 0], 'repair': [.5, .5, .5]}]
    result = group_statistics(pairs, config)
    assert result['all']['n'] == 3
    assert result['ambiguous']['n'] == 1
    assert result['static']['repair']['mae'] == pytest.approx(.15)
    assert result['static']['repair']['delta'] == pytest.approx(.05)
    assert result['static']['repair']['cf_count'] == 2
    assert result['motion']['repair']['mae'] == pytest.approx(.1)
    assert result['all']['repair']['base'] == pytest.approx(.4)


def test_protocol_keeps_model_and_intervention_frozen():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    bmc = json.loads((root / 'configs/dynamic-generalization/bmc-v1.json').read_text())
    old = json.loads((root / 'configs/dynamic-generalization/lasiesta-v1.json').read_text())
    for k in ('head_sha256', 'head_implementation_sha256', 'probe_config_sha256', 'raft_sha256', 'origin_upstream_commit', 'views', 'intervention'):
        assert bmc[k] == old[k]
    assert bmc['scope']['training_updates'] == 0


@pytest.mark.parametrize('source_fps', [25, 7])
def test_prepare_decode_and_review_freeze_end_to_end(tmp_path, monkeypatch, source_fps):
    import cv2
    import shutil
    import sys
    from types import SimpleNamespace
    from scripts.counterfactual.evaluate_dynamic_bmc import digest, freeze, prepare
    try:
        import imageio_ffmpeg
    except ImportError:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            pytest.skip('A local FFmpeg is required for decode integration')
        monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', SimpleNamespace(get_ffmpeg_exe=lambda: ffmpeg))
    video = tmp_path / 'Video_001.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), source_fps, (96, 64))
    assert writer.isOpened()
    for i in range(100):
        writer.write(np.full((64, 96, 3), i, np.uint8))
    writer.release()
    download = tmp_path / 'download'; download.mkdir()
    archive = download / 'Video_001.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.write(video, 'Video_001/Video_001.avi')
    (download / 'completion.json').write_text(json.dumps({'status': 'finished', 'sequences': [
        {'sequence': 'Video_001', 'sha256': digest(str(archive))}]}))
    config = {'sequences': ['Video_001'], 'sampling': {'max_windows_per_sequence': 12, 'timestamp_tolerance': 1e-4},
              'labels': {'conditions': ['static', 'motion', 'ambiguous']}}
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    selection = tmp_path / 'selection'
    args = SimpleNamespace(output=str(selection), config=str(config_path), download=str(download))
    prepare(args, config)
    candidates = [json.loads(s) for s in (selection / 'candidates.jsonl').read_text().splitlines()]
    if source_fps < 8:
        assert candidates == []
        originals = json.loads((selection / 'originals.json').read_text())
        assert originals[0]['status'] == 'NOT SCORED' and originals[0]['source_fps'] == 7
        return
    assert len(candidates) == 2 and all(len(r['source_frames']) == 16 for r in candidates)
    review = {'scores_read': False, 'official_ground_truth': False, 'clips': {
        r['source_id']: {'condition': 'ambiguous', 'reason': 'Test fixture, not research data'} for r in candidates}}
    review_path = tmp_path / 'review.json'; review_path.write_text(json.dumps(review))
    freeze(SimpleNamespace(selection=str(selection), config=str(config_path), review=str(review_path)), config)
    receipt = json.loads((selection / 'selection.json').read_text())
    assert receipt['clips'] == 2 and receipt['class_counts'] == {'ambiguous': 2}
    assert receipt['sources_sha256'] == digest(selection / 'sources.jsonl')


def test_rejected_timestamp_probe_drains_child_without_pipe_deadlock(tmp_path, monkeypatch):
    import io
    import sys
    from types import SimpleNamespace
    from scripts.counterfactual import evaluate_dynamic_bmc as module
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', SimpleNamespace(get_ffmpeg_exe=lambda: __file__))
    class Child:
        def __init__(self):
            self.stderr = io.StringIO('config in time_base: 1/25\nn: 0 pts: 0 pts_time: 0\nn: 1 pts: 3 pts_time: 0.12\n')
            self.returncode = None; self.drained = False
        def poll(self):
            return self.returncode
        def terminate(self):
            pass
        def communicate(self, timeout=None):
            self.drained = True; self.returncode = -15
            return (None, '')
    child = Child()
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: child)
    with pytest.raises(ValueError, match='non-CFR'):
        module.ffmpeg_timeline('unused.avi', tmp_path / 'timeline.log', 25, 1e-4)
    assert child.drained
