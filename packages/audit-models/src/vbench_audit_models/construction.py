"""Independent construction mask models. Never consumed by scoring backends."""
from __future__ import annotations

import numpy as np
from vbench_audit_core.inputs import sha256_file


class ConstructionMasks:
    def __init__(self, segformer_dir, sam_checkpoint, sam_source, *, device):
        import sys
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        from vbench_audit_models.foreground import load_mobile_sam_predictor
        self.device = device
        self.processor = SegformerImageProcessor.from_pretrained(str(segformer_dir), local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(str(segformer_dir), local_files_only=True).to(device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        if str(sam_source) not in sys.path:
            sys.path.insert(0, str(sam_source))
        self.sam = load_mobile_sam_predictor(sam_checkpoint, device)
        self.provenance = {"role": "construction_only", "scoring_reuse": False,
            "segformer_files_sha256": {p.name: sha256_file(p) for p in sorted(segformer_dir.iterdir()) if p.is_file()},
            "mobile_sam_sha256": sha256_file(sam_checkpoint), "sam_point_source": "independent target box center",
            "segformer_use_gate": "exact label inside independent target box and mask/box area >=0.2",
            "fallback": "MobileSAM with independent box and center foreground point", "feather": "inward 1.5px"}

    def for_frames(self, frames, target, target_boxes):
        import torch
        masks, evidence = [], []
        ids = [idx for idx, name in self.id2label.items() if name.casefold() == target.casefold()]
        for image, boxes in zip(frames, target_boxes):
            if not boxes:
                raise ValueError("construction requires independent target localization on every sampled frame")
            values = self.processor(images=image, return_tensors="pt", size={"height": 512, "width": 512})
            with torch.inference_mode():
                logits = self.model(**{k: v.to(self.device) for k, v in values.items()}).logits
                semantic = torch.nn.functional.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
            union, sources = np.zeros(image.shape[:2], np.uint8), []
            self.sam.set_image(np.ascontiguousarray(image), image_format="RGB")
            for detection in boxes:
                x0, y0, x1, y1 = detection["box"]
                region = np.zeros(image.shape[:2], np.uint8)
                region[max(0, int(y0)):min(image.shape[0], int(np.ceil(y1))), max(0, int(x0)):min(image.shape[1], int(np.ceil(x1)))] = 1
                mask = (np.isin(semantic, ids) & (region == 1)).astype(np.uint8)
                if mask.sum() >= .2 * region.sum() and mask.any():
                    source = "segformer_exact_class_inside_independent_box"
                else:
                    prediction, quality, _ = self.sam.predict(box=np.asarray([x0, y0, x1, y1]),
                        point_coords=np.asarray([[(x0+x1)/2, (y0+y1)/2]]), point_labels=np.asarray([1]), multimask_output=False)
                    mask = (np.asarray(prediction[0], dtype=np.uint8) & region)
                    source = "mobilesam_box_plus_center_point"
                if not mask.any():
                    raise ValueError("empty construction mask")
                union |= mask
                sources.append({"locator": detection, "source": source})
            masks.append(union)
            evidence.append({"area_fraction": float(union.mean()), "instances": sources})
        return np.stack(masks), evidence
