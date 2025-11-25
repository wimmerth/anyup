import os
import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import InterpolationMode
from typing import Callable, Optional, Dict, Any, List
from torch.utils.data.dataloader import default_collate

from .dataloader.post_collate import BatchMultiCrop


def seed_worker():
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def logger(args, base_log_dir):
    os.makedirs(base_log_dir, exist_ok=True)
    existing_versions = [
        int(d.split("_")[-1])
        for d in os.listdir(base_log_dir)
        if os.path.isdir(os.path.join(base_log_dir, d)) and d.startswith("version_")
    ]
    new_version = max(existing_versions, default=-1) + 1
    new_log_dir = os.path.join(base_log_dir, f"version_{new_version}")

    # Create the SummaryWriter with the new log directory
    writer = SummaryWriter(log_dir=new_log_dir)
    return writer, new_version, new_log_dir


def build_image_transform(img_size: int, mean, std, center_crop: bool = True):
    ops: List[T.Transform] = [
        T.ToImage(),
        T.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ]
    if center_crop:
        ops.append(T.CenterCrop((img_size, img_size)))
    ops.extend([T.ToDtype(torch.float32, scale=True), T.Normalize(mean=mean, std=std)])
    return T.Compose(ops)


def build_augmentation_transform(img_size: int, mean, std, center_crop: bool = True, strength: float = 1):
    ops: List[T.Transform] = [
        T.ToImage(),
        T.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
    ]
    if center_crop:
        ops.append(T.CenterCrop((img_size, img_size)))
    augmentation_transforms = [
        T.ToDtype(torch.uint8, scale=True),  # work in uint8 for color/JPEG
        T.RandomApply([
            T.ColorJitter(0.2, 0.2, 0.3, 0.05),
            T.RandomPhotometricDistort(p=0.5),
        ], p=0.5 ** (1 / strength)),
        T.RandomGrayscale(p=0.1 ** (1 / strength)),

        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.3, 1.5))], p=0.35 ** (1 / strength)),
        T.RandomChoice([  # JPEG with a few discrete qualities
            T.JPEG(95), T.JPEG(80), T.JPEG(60), T.JPEG(40)
        ], p=[0.25, 0.35, 0.25, 0.15], ),
        T.RandomApply([T.RandomAutocontrast()], p=0.2 ** (1 / strength)),
        T.RandomApply([T.RandomEqualize()], p=0.15 ** (1 / strength)),
        T.RandomApply([T.RandomAdjustSharpness(1.5)], p=0.2 ** (1 / strength)),
        T.RandomApply([T.RandomPosterize(bits=5)], p=0.1 ** (1 / strength)),
        T.RandomApply([T.RandomSolarize(threshold=0.9)], p=0.05 ** (1 / strength)),
        T.RandomApply([T.RandomInvert()], p=0.02 ** (1 / strength)),

        T.ToDtype(torch.float32, scale=True),  # now in [0,1] float
        T.RandomApply([T.GaussianNoise(sigma=0.03, clip=True)], p=0.6 ** (1 / strength)),
    ]
    ops.extend(augmentation_transforms)
    ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)


def build_label_transform(target_size: int, is_eval: bool):
    if not is_eval:
        return None
    return T.Compose([
        T.Resize(target_size, interpolation=InterpolationMode.NEAREST),
        T.CenterCrop((target_size, target_size)),
        T.PILToTensor()
    ])


def get_dataloaders(cfg, backbone, is_evaluation=False, mean=None, std=None, shuffle=True, augmentation_strength=1.0):
    # Default ImageNet stats
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    if mean is None:
        mean = getattr(backbone, "config", {}).get("mean", default_mean)
    if std is None:
        std = getattr(backbone, "config", {}).get("std", default_std)

    # sample-level transforms
    crop_size = cfg.img_size if not "crop_size" in cfg else cfg.crop_size
    print(f"crop_size={crop_size!r}  type={type(crop_size)}")
    image_tf = build_image_transform(crop_size, mean, std, center_crop=True)
    augmentation_tf = build_augmentation_transform(crop_size, mean, std, center_crop=True,
                                                   strength=augmentation_strength)

    # datasets
    if is_evaluation:
        label_tf = build_label_transform(cfg.target_size, is_eval=is_evaluation)
        train_dataset = instantiate(cfg.dataset_evaluation, _convert_="partial",
                                    transform=image_tf, target_transform=label_tf)
        val_dataset = instantiate(cfg.dataset_evaluation, _convert_="partial",
                                  transform=image_tf, target_transform=label_tf, split="val")
    else:
        train_dataset = instantiate(cfg.train_dataset, _convert_="partial", transform=image_tf,
                                    augmentation_transform=augmentation_tf)
        val_dataset = instantiate(cfg.val_dataset, _convert_="partial", transform=image_tf)

    # batch-level transforms via collate_fn
    train_batch_tf = None
    val_batch_tf = None
    if not is_evaluation:
        global_view_random_resize = getattr(cfg, "global_view_random_resize", None)
        if isinstance(global_view_random_resize, float):
            global_view_random_resize = (global_view_random_resize, 1.0)
        elif isinstance(global_view_random_resize, str):
            from ast import literal_eval as make_tuple
            global_view_random_resize = make_tuple(global_view_random_resize)
        elif global_view_random_resize is not None:
            raise ValueError("global_view_random_resize must be float or tuple(float, float)")

        train_batch_tf = BatchMultiCrop(
            crop_size=cfg.img_size, patch_size=backbone.patch_size, num_crops=cfg.num_local_crops,
            global_view_random_resize=global_view_random_resize
        )
        val_batch_tf = BatchMultiCrop(
            crop_size=cfg.img_size, patch_size=backbone.patch_size, num_crops=cfg.num_local_crops,
            global_view_random_resize=global_view_random_resize
        )


    g = torch.Generator()
    if shuffle:
        g.manual_seed(0)  # reproducible shuffling

    train_loader = instantiate(
        cfg.train_dataloader,
        dataset=train_dataset,
        generator=g,
        collate_fn=make_collate_fn(train_batch_tf),
    )
    val_loader = instantiate(
        cfg.val_dataloader,
        dataset=val_dataset,
        generator=g,
        collate_fn=make_collate_fn(val_batch_tf),
    )
    return train_loader, val_loader


def get_batch(batch, device):
    """Process batch and return required tensors."""
    for key in batch:
        if key.endswith("image"):
            batch[key] = batch[key].to(device)
    return batch


def make_collate_fn(batch_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
    """
    Wrap default_collate and then run an optional batch_transform on the collated dict.
    """

    def _collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = default_collate(samples)  # stacks "image" to (B,C,H,W) automatically
        if batch_transform is not None:
            batch = batch_transform(batch)
        return batch

    return _collate_fn
