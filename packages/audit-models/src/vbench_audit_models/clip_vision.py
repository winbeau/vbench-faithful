"""Load the original CLIP ViT image tower without initializing unused text weights."""
from pathlib import Path

import torch


class CLIPImageEncoder(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def encode_image(self, images):
        return self.visual(images.to(self.visual.conv1.weight.dtype))


def load_clip_vision(checkpoint, device):
    """Use CLIP's own architecture/casts and strictly copy every visual weight.

    Only the verified local JIT ViT checkpoints used by the evaluator are accepted.
    Meta construction skips random initialization. Copying (rather than assigning)
    checkpoint tensors preserves CLIP's mixed parameter dtypes, including FP32
    layer norms/embeddings and FP16 convolution/attention/projection weights.
    """
    from clip.model import VisionTransformer, convert_weights

    state = torch.jit.load(str(Path(checkpoint)), map_location="cpu").state_dict()
    visual_state = {k.removeprefix("visual."): v for k, v in state.items() if k.startswith("visual.")}
    if "proj" not in visual_state:
        raise ValueError("CLIP image-only loading requires a ViT checkpoint")
    width = visual_state["conv1.weight"].shape[0]
    patch_size = visual_state["conv1.weight"].shape[-1]
    patches = visual_state["positional_embedding"].shape[0] - 1
    grid = round(patches ** .5)
    if grid * grid != patches:
        raise ValueError("CLIP checkpoint has a non-square positional embedding")
    layers = sum(k.endswith(".attn.in_proj_weight") for k in visual_state)
    with torch.device("meta"):
        visual = VisionTransformer(grid * patch_size, patch_size, width, layers,
                                   width // 64, visual_state["proj"].shape[1])
        convert_weights(visual)
    visual.to_empty(device=device)
    visual.load_state_dict(visual_state, strict=True)
    if torch.device(device).type == "cpu":
        visual.float()
    return CLIPImageEncoder(visual).eval()
