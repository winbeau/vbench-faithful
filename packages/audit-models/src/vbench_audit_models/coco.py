"""Full COCO instance locator, independent of GRiT; no scoring predicates."""
from __future__ import annotations

from .foreground import CocoSubjectBoxDetector


class CocoInstanceLocator(CocoSubjectBoxDetector):
    def __init__(self, checkpoint, *, device, expected_sha256, threshold=.7):
        super().__init__(checkpoint, expected_sha256=expected_sha256, device=device,
                         threshold=threshold, size=512)
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
        self.categories = list(MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"])
        self.provenance.update({"classes": self.categories, "role": "independent_construction_locator",
                                "grit_used": False})

    def instances_for(self, frames):
        import torch
        result = []
        with torch.inference_mode():
            for start in range(0, len(frames), 4):
                images = [torch.as_tensor(f.copy()).permute(2, 0, 1).to(self.device, dtype=torch.float32)/255
                          for f in frames[start:start+4]]
                for prediction in self.model(images):
                    keep = prediction["scores"] >= self.threshold
                    result.append([{"box": box, "score": score, "label": self.categories[label], "label_id": label}
                        for box, score, label in zip(prediction["boxes"][keep].cpu().tolist(),
                           prediction["scores"][keep].cpu().tolist(), prediction["labels"][keep].cpu().tolist())])
        return result
