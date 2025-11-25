from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torchvision.transforms.v2.functional as Tv2F
import random


def _round_to_multiple(v: int, m: int) -> int:
    return int(m * round(v / m))


class BatchTransform:
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class BatchMultiCrop(BatchTransform):
    """
    For each image in the batch, sample K patch-aligned crops.
    Now supports dynamic patch sizes for multi-backbone training.
    """

    def __init__(self, crop_size: int,
                 patch_size: Union[int, List[int]],
                 num_crops: int = 4,
                 global_view_random_resize: Optional[Tuple[float, float]] = None):
        self.crop_size = crop_size
        # Store unique patch sizes sorted
        if isinstance(patch_size, int):
            self.patch_sizes = [patch_size]
        else:
            self.patch_sizes = sorted(list(set(patch_size)))

        self.num_crops = num_crops
        self.global_view_random_resize = global_view_random_resize

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        img = batch.pop("image")  # (B,C,H,W)
        if img.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(img.shape)}")

        current_patch_size = random.choice(self.patch_sizes)

        B, C, H, W = img.shape
        S = self.crop_size

        max_top = max(0, H - S)
        max_left = max(0, W - S)

        crops = []
        coords = []  # (row, col) in patch units

        for b in range(B):
            for _ in range(self.num_crops):
                # patch-aligned random top/left using current_patch_size
                top = current_patch_size * torch.randint(0, (max_top // current_patch_size) + 1, (1,)).item()
                left = current_patch_size * torch.randint(0, (max_left // current_patch_size) + 1, (1,)).item()

                crops.append(Tv2F.crop(img[b], top=top, left=left, height=S, width=S))
                # Coordinates are stored relative to the current patch unit
                coords.append((top // current_patch_size, left // current_patch_size))

        hr = torch.stack(crops, dim=0)  # (B*K,C,S,S)
        lr = Tv2F.resize(img, size=[S, S], antialias=True)  # (B,C,S,S)
        augmented_img = batch.pop("aug_image")
        augmented_guidance = Tv2F.resize(augmented_img, size=[S, S], antialias=True)

        if self.global_view_random_resize is not None:
            scale_min, scale_max = self.global_view_random_resize
            scale = torch.empty(1).uniform_(scale_min, scale_max).item()
            # Align resizing to current_patch_size
            new_H = max(current_patch_size, _round_to_multiple(int(H * scale), current_patch_size))
            new_W = max(current_patch_size, _round_to_multiple(int(W * scale), current_patch_size))
            lr = Tv2F.resize(lr, size=[new_H, new_W], antialias=True)
            augmented_guidance = Tv2F.resize(augmented_guidance, size=[new_H, new_W], antialias=True)

        batch["hr_image"] = hr
        batch["guidance_image"] = lr
        batch["augmented_guidance_image"] = augmented_guidance
        batch["lr_image"] = lr
        batch["guidance_crop"] = torch.tensor(coords, dtype=torch.long)  # (B*K,2)

        # Tell the training loop which patch size was used
        batch["patch_size"] = current_patch_size
        batch["upsampling_size"] = H // current_patch_size

        return batch
