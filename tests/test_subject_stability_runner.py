from types import SimpleNamespace

import torch

from scripts.counterfactual.run_subject_stability import upstream_score_from_frames


def load_video(path):
    raise AssertionError('The reference harness must not decode a different clip')


def fake_upstream(model, paths, device, read_frame):
    frames=load_video(paths[0])
    assert frames.dtype == torch.float32 and not read_frame
    return None,[{'video_results':model(frames).item()}]


def test_upstream_reference_receives_current_pixels_without_mutating_globals():
    original=fake_upstream.__globals__['load_video']
    frames=torch.full((2,3,4,4),123,dtype=torch.uint8)
    value=upstream_score_from_frames(SimpleNamespace(subject_consistency=fake_upstream),
                                    lambda x:x.mean()/255,frames,'cpu')
    assert abs(value-123/255)<1e-7
    assert fake_upstream.__globals__['load_video'] is original
