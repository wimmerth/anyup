dependencies = ['torch']

from anyup.model import AnyUp
import torch


def anyup(use_natten=False, pretrained: bool = True, device='cpu'):
    """
    AnyUp model trained on DINOv2 ViT-S/14 features, used in most experiments of the paper.
    Note: If you want to use vis_attn, you also need to install matplotlib. If you want to use NATTEN, you need to
    install the compatible natten version for your system.
    """
    model = AnyUp().to(device)
    if pretrained:
        checkpoint = "https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth"
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location=device))
    if use_natten:
        from anyup.layers import setup_cross_attention_block
        model.cross_decode = setup_cross_attention_block(
            use_natten=True,
            qk_dim=model.cross_decode.cross_attn.attention.embed_dim,
            num_heads=4,
            window_ratio=model.cross_decode.window_ratio,
            use_params_from=model.cross_decode,
        ).to(device)
    return model


def anyup_multi_backbone(use_natten=False, pretrained: bool = True, device='cpu'):
    """
    AnyUp model trained on features from multiple backbones (DINOv2 (S), DINOv2-R (S), CLIP (B), SigLIP (B), ViT-B).
    Note: If you want to use vis_attn, you also need to install matplotlib. If you want to use NATTEN, you need to
    install the compatible natten version for your system.
    """
    model = AnyUp().to(device)
    if pretrained:
        checkpoint = "https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth"
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location=device))
    if use_natten:
        from anyup.layers import setup_cross_attention_block
        model.cross_decode = setup_cross_attention_block(
            use_natten=True,
            qk_dim=model.cross_decode.cross_attn.attention.embed_dim,
            num_heads=4,
            window_ratio=model.cross_decode.window_ratio,
            use_params_from=model.cross_decode,
        ).to(device)
    return model
