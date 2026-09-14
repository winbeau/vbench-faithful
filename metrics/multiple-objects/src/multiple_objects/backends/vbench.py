from __future__ import annotations

import importlib
import json
import tempfile
import types
from pathlib import Path
from typing import Any, Callable, Mapping

from vbench_audit_core.upstream import (
    import_official_module as _import_official_module,
    inspect_upstream as _inspect_upstream,
)

from ..conditions import parse_target_objects
from ..schemas import DetectionEvidence, FrameDetections

UPSTREAM_PATH = None
DEFAULT_WEIGHT = Path.home() / ".cache/vbench/grit_model/grit_b_densecap_objectdet.pth"
OFFICIAL_THRESHOLD = 0.5
OFFICIAL_NUM_FRAMES = 16


def inspect_upstream(path: Path | None = None):
    return _inspect_upstream(path)


def import_official_module(path: Path | None = None):
    return _import_official_module("multiple_objects", path)


def metadata_to_official_entries(videos: list[Path], metadata: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for video in videos:
        item = metadata[video.name]
        targets = parse_target_objects(item)
        if len(targets) != 2:
            raise ValueError("official VBench multiple_objects requires exactly two ' and '-separated targets")
        entries.append({"prompt_en": item.get("prompt", ""), "dimension": ["multiple_objects"], "video_list": [str(video.resolve())], "auxiliary_info": {"multiple_objects": {"object": " and ".join(targets)}}})
    return entries


def evaluate_official_reference(videos: list[Path], metadata: Mapping[str, Mapping[str, Any]], device: Any, model_weight: Path, *, compute: Callable[..., Any] | None = None, upstream_path: Path | None = None) -> tuple[Any, dict[str, Any] | None]:
    if compute is None and not model_weight.is_file():
        raise FileNotFoundError(f"official GRiT weight not found: {model_weight}")
    state = None
    if compute is None:
        module, state = import_official_module(upstream_path)
        compute = module.compute_multiple_objects
    with tempfile.TemporaryDirectory(prefix="multiple-objects-official-") as root:
        path = Path(root) / "official_input.json"
        path.write_text(json.dumps(metadata_to_official_entries(videos, metadata), ensure_ascii=False), encoding="utf-8")
        raw = compute(str(path), device, {"model_weight": str(model_weight)})
    return raw, state


def evaluate_official_batch(
    videos: list[Path],
    metadata: Mapping[str, Mapping[str, Any]],
    device: Any,
    model_weight: Path,
) -> list[dict[str, Any]]:
    evaluator = OfficialVBenchEvaluator(device, model_weight)
    results: list[dict[str, Any]] = []
    for video in videos:
        try:
            item = evaluator.evaluate_video(video, metadata[video.name])
            results.append(
                {
                    "video": str(video),
                    "prompt": str(metadata[video.name].get("prompt", "")),
                    "backend": "vbench",
                    "score": float(item["video_results"]),
                    "status": "succeeded",
                    "diagnostics": {
                        "success_frame_count": int(item["success_frame_count"]),
                        "frame_count": int(item["frame_count"]),
                        "frame_score_sum": float(item["success_frame_count"]),
                        "official_threshold": OFFICIAL_THRESHOLD,
                    },
                }
            )
        except Exception as exc:
            results.append(
                {
                    "video": str(video),
                    "prompt": str(metadata[video.name].get("prompt", "")),
                    "backend": "vbench",
                    "score": None,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "diagnostics": None,
                }
            )
    return results


class OfficialGrITDetector:
    """GRiT ObjectDet adapter retaining aligned labels, boxes and final scores.

    Repair inference lowers only the same ROI candidate threshold configured by
    upstream. Instrumentation retains the exact score before GRiT overwrites
    ``Instances.scores`` with a description-adjusted final confidence.
    """

    def __init__(
        self,
        device: Any,
        model_weight: Path,
        upstream_path: Path | None = None,
        *,
        repair_candidate_threshold: float = 0.0,
    ):
        if not model_weight.is_file():
            raise FileNotFoundError(f"official GRiT weight not found: {model_weight}")
        module, state = import_official_module(upstream_path)
        self.module = module
        self.upstream_state = state
        self.device = device
        threshold = float(repair_candidate_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("repair_candidate_threshold must be in [0, 1]")
        image_api = importlib.import_module("vbench.third_party.grit_src.image_dense_captions")
        selected_root = Path(state.path)
        if selected_root not in Path(image_api.__file__).resolve().parents:
            raise RuntimeError(f"wrong GRiT helper import: {image_api.__file__}")
        args = image_api.get_parser(device, str(model_weight))
        args["test_task"] = "ObjectDet"
        args["confidence_threshold"] = threshold
        self.model = module.DenseCaptioning(device)
        self.model.demo = image_api.VisualizationDemo(image_api.setup_cfg(args))
        self.repair_candidate_threshold = threshold
        self._last_object_threshold_scores = None
        self._pending_primary_selection = None
        self._instrument_official_threshold_scores()

    def _instrument_official_threshold_scores(self) -> None:
        """Attach the threshold input score without changing upstream outputs."""
        roi_heads = self.model.demo.predictor.model.roi_heads
        original_inference = roi_heads.fast_rcnn_inference_GRiT

        def instrumented_inference(_roi_heads: Any, *args: Any, **kwargs: Any):
            instances, kept_indices = original_inference(*args, **kwargs)
            for instance in instances:
                # At this point threshold, NMS and top-k have selected instances,
                # but the text decoder has not yet overwritten instance.scores.
                instance.official_threshold_scores = instance.scores.detach().clone()
            selection = tuple(
                (
                    instance.pred_boxes.tensor.detach().clone(),
                    instance.official_threshold_scores.detach().clone(),
                )
                for instance in instances
            )
            if self._pending_primary_selection is None:
                self._pending_primary_selection = selection
            else:
                primary = self._pending_primary_selection
                if len(primary) != len(selection) or any(
                    not first_boxes.equal(second_boxes)
                    or not first_scores.equal(second_scores)
                    for (first_boxes, first_scores), (second_boxes, second_scores)
                    in zip(primary, selection)
                ):
                    raise RuntimeError(
                        "GRiT primary/ObjectDet candidate selections are not index aligned"
                    )
                self._pending_primary_selection = None
            return instances, kept_indices

        roi_heads.fast_rcnn_inference_GRiT = types.MethodType(
            instrumented_inference, roi_heads
        )
        original_forward_object = roi_heads.forward_object

        def instrumented_forward_object(_roi_heads: Any, *args: Any, **kwargs: Any):
            result = original_forward_object(*args, **kwargs)
            instances, _ = result
            if len(instances) != 1 or not instances[0].has("official_threshold_scores"):
                raise RuntimeError("GRiT ObjectDet threshold scores were not captured")
            self._last_object_threshold_scores = instances[0].official_threshold_scores
            return result

        roi_heads.forward_object = types.MethodType(instrumented_forward_object, roi_heads)

    def _set_roi_threshold(self, threshold: float) -> None:
        predictors = self.model.demo.predictor.model.roi_heads.box_predictor
        if not isinstance(predictors, (list, tuple)):
            try:
                predictors = list(predictors)
            except TypeError:
                predictors = [predictors]
        if not predictors:
            raise RuntimeError("GRiT ROI box predictor is empty")
        for predictor in predictors:
            predictor.test_score_thresh = threshold

    @staticmethod
    def _extract(
        predictions: Mapping[str, Any], threshold_scores: Any
    ) -> tuple[DetectionEvidence, ...]:
        instances = predictions.get("instances")
        if instances is None or not instances.has("det_obj"):
            raise RuntimeError("GRiT prediction lacks official ObjectDet labels")
        labels = list(instances.det_obj.data)
        if not instances.has("pred_boxes") or not instances.has("scores"):
            raise RuntimeError("GRiT prediction lacks boxes or scores")
        boxes = instances.pred_boxes.tensor.detach().cpu().tolist()
        final_scores = instances.scores.detach().cpu().tolist()
        if threshold_scores is None:
            raise RuntimeError("GRiT official threshold score capture is missing")
        official_scores = threshold_scores.detach().cpu().tolist()
        if not (
            len(labels) == len(boxes) == len(final_scores)
            == len(official_scores) == len(instances)
        ):
            raise RuntimeError("GRiT label/box/threshold-score fields are not aligned")
        return tuple(
            DetectionEvidence(
                str(label), float(official_score), tuple(box), float(final_score)
            )
            for label, box, official_score, final_score in zip(
                labels, boxes, official_scores, final_scores
            )
        )

    def detect_video(self, video: Path) -> list[FrameDetections]:
        video_tensor = self.module.load_video(str(video), num_frames=OFFICIAL_NUM_FRAMES)
        _, _, height, width = video_tensor.size()
        if min(height, width) > 768:
            scale = 720.0 / min(height, width)
            video_tensor = self.module.transforms.Resize(size=(int(scale * height), int(scale * width)))(video_tensor)
        output: list[FrameDetections] = []
        for frame in video_tensor.permute(0, 2, 3, 1):
            self._set_roi_threshold(self.repair_candidate_threshold)
            repair_predictions, _ = self.model.run_det_tensor(frame.detach().cpu().numpy())
            detections = self._extract(
                repair_predictions, self._last_object_threshold_scores
            )
            self._set_roi_threshold(OFFICIAL_THRESHOLD)
            official_predictions, _ = self.model.run_det_tensor(frame.detach().cpu().numpy())
            official_labels = tuple(
                item.label
                for item in self._extract(
                    official_predictions, self._last_object_threshold_scores
                )
            )
            output.append(FrameDetections(detections, official_labels))
        self._set_roi_threshold(self.repair_candidate_threshold)
        return output


class OfficialVBenchEvaluator:
    """Cached official multiple-object evaluator for per-video isolation."""

    def __init__(self, device: Any, model_weight: Path, upstream_path: Path | None = None):
        if not model_weight.is_file():
            raise FileNotFoundError(f"official GRiT weight not found: {model_weight}")
        self.module, self.upstream_state = import_official_module(upstream_path)
        self.device = device
        self.model = self.module.DenseCaptioning(device)
        self.model.initialize_model_det(model_weight=str(model_weight))

    def evaluate_video(self, video: Path, metadata_item: Mapping[str, Any]) -> Mapping[str, Any]:
        targets = parse_target_objects(metadata_item)
        if len(targets) != 2:
            raise ValueError("official VBench multiple_objects requires exactly two target objects")
        entry = {
            "prompt": metadata_item.get("prompt", ""),
            "video_list": [str(video.resolve())],
            "auxiliary_info": {"object": " and ".join(targets)},
        }
        _, rows = self.module.multiple_objects(self.model, [entry], self.device)
        return rows[0]
