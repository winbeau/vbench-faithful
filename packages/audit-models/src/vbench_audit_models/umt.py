"""Pinned UMT initialization shared by the paper evidence workers."""
import json


def initialize_umt(args, device):
    # Import the pinned official module: it registers the exact timm architecture.
    import torch
    from vbench import human_action as upstream
    model = upstream.create_model('vit_large_patch16_224', pretrained=False, num_classes=400,
        all_frames=16, tubelet_size=1, use_learnable_pos_emb=False, fc_drop_rate=0., drop_rate=0.,
        drop_path_rate=0.2, attn_drop_rate=0., drop_block_rate=None, use_checkpoint=False,
        checkpoint_num=16, use_mean_pooling=True, init_scale=0.001)
    state = torch.load(args.umt_weights, map_location='cpu')
    loading = model.load_state_dict(state, strict=False)
    transform = upstream.Compose([upstream.Resize(256, interpolation='bilinear'),
        upstream.CenterCrop(size=(224, 224)), upstream.ClipToTensor(),
        upstream.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    print(json.dumps({'umt_missing_keys': loading.missing_keys, 'umt_unexpected_keys': loading.unexpected_keys}), flush=True)
    return model.to(device).eval(), transform, upstream.build_dict()

