from pathlib import Path

import numpy as np
import pytest
import torch

from background_consistency import runtime
from background_consistency.algorithms import score_views
from background_consistency.models import OfficialClipEncoder
from vbench_audit_models.foreground import MobileSamForegroundProvider


class FakeEncoder:
    provenance = {"test": True}

    def decode(self, path):
        if path.name == "bad.mp4":
            raise ValueError("unreadable video")
        return torch.ones((3, 3, 8, 8), dtype=torch.uint8)

    def features(self, frames):
        return frames.float().mean((2, 3))


def test_batch_failures_keep_the_video_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "build_model", lambda *a, **kw: FakeEncoder())
    videos = [tmp_path / name for name in ("good.mp4", "bad.mp4")]
    for p in videos:
        p.touch()
    result = runtime.evaluate_videos("vbench", videos, {}, "cpu", {"model": {"clip": {"checkpoint": "test"}}}, method="official")
    assert [r.status for r in result] == ["succeeded", "failed"]
    assert result[0].score == pytest.approx(1)
    assert result[1].score is None and "unreadable" in result[1].error


def test_adapter_casts_rgb_to_float_before_the_official_resize():
    encoder = object.__new__(OfficialClipEncoder)
    encoder.device = "cpu"
    class Model:
        def encode_image(self, images):
            return images.mean((2, 3))
    encoder.model = Model()
    def transform(frames):
        assert frames.dtype == torch.float32
        assert frames.max() == 254
        return frames / 255
    encoder.transform = transform
    frames = torch.full((2, 3, 8, 8), 254, dtype=torch.uint8)
    torch.testing.assert_close(encoder.features(frames), encoder.features(frames.float()))


def test_masked_views_are_identical_when_only_predicted_foreground_changes():
    frames = torch.arange(3*3*8*8).reshape(3, 3, 8, 8).to(torch.uint8)
    masks = np.zeros((3, 8, 8), np.uint8)
    masks[:, 2:6, 2:6] = 1
    changed = frames.clone()
    changed[:, :, 2:6, 2:6] = 0
    a, fa = runtime.encode_views(FakeEncoder(), frames, masks)
    b, fb = runtime.encode_views(FakeEncoder(), changed, masks)
    for view in ("frame", "union"):
        assert torch.equal(a[view], b[view])
    assert not torch.equal(a["global"], b["global"])
    assert score_views(a, fa)["union_official"] == score_views(b, fb)["union_official"]


def test_independent_foreground_provider_does_not_reuse_clean_masks():
    class Detector:
        provenance = {"test": True}
        def boxes_for(self, frames, phrase):
            assert phrase is None
            return [{"boxes": [[0, 0, 2, 2]] if int(f[0, 0, 0]) else [], "scores": []} for f in frames]
    class Predictor:
        def set_image(self, image, image_format):
            self.shape = image.shape[:2]
        def predict(self, box, multimask_output):
            result = np.zeros((1, *self.shape), bool)
            result[:, :2, :2] = 1
            return result, np.array([.9]), None
    provider = MobileSamForegroundProvider(Detector(), Predictor(), weights_sha256="test")
    assert provider.masks_for(torch.ones((2, 3, 4, 4))).sum() == 8
    assert provider.masks_for(torch.zeros((2, 3, 4, 4))).sum() == 0
    assert provider.last_diagnostics["num_empty_foreground_frames"] == 2
    assert provider.provenance["construction_masks_reused"] is False


def test_calibrated_patch_entry_uses_new_independent_masks_and_keeps_failure(tmp_path, monkeypatch):
    from background_consistency import patches
    from background_consistency.calibration import patch_score_grid
    calls = []
    class Provider:
        provenance = {'role': 'scoring'}
        last_diagnostics = {'test': True}
        def masks_for(self, frames):
            calls.append('localize')
            return np.zeros((len(frames), 8, 8), np.uint8)
    features = torch.tensor([[1., 0.], [.8, .6], [.8, .6]])
    views = {k: features for k in ('global', 'frame', 'union')}
    fractions = {k: torch.ones(3) for k in views}
    valid = {k: torch.ones(3, dtype=torch.bool) for k in views}
    def patch_view(encoder, frames, foreground):
        assert calls == ['localize']
        assert foreground.shape == (3, 8, 8)
        return features, None, views, fractions, valid
    monkeypatch.setattr(runtime, 'build_model', lambda *a, **kw: FakeEncoder())
    monkeypatch.setattr(runtime, 'build_foreground_provider', lambda *a, **kw: Provider())
    monkeypatch.setattr(patches, 'patch_views', patch_view)
    videos = [tmp_path/'good.mp4', tmp_path/'bad.mp4']
    for path in videos:
        path.touch()
    rows = runtime.evaluate_videos('audit', videos, {}, 'cpu',
        {'model': {'clip': {'checkpoint': 'test'}, 'localizer': {}}}, method='patch_frame_calibrated')
    expected = patch_score_grid(features, views, valid)['patch_frame_all_pairs_calibrated']
    assert rows[0].score == expected
    assert rows[0].metric['patch_gain'] == 1.75
    assert rows[1].status == 'failed'
