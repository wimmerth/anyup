# Adapted from JAFAR (https://github.com/PaulCouairon/JAFAR)
import datetime
import os
import random
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
from rich import print
from rich.console import Console
from rich.syntax import Syntax
from torch import autocast
from tqdm import tqdm

from anyup.utils.training import get_batch, get_dataloaders, logger
from anyup.loss import Cosine_MSE

FREQ = 100


def round_to_nearest_multiple(value, multiple=14):
    return multiple * round(value / multiple)


@hydra.main(config_path="config", config_name="base")
def trainer(cfg: DictConfig):
    yaml_syntax = Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="monokai", line_numbers=True)
    print(yaml_syntax)
    seed = 0
    print(f"Seed: {seed}")
    torch.manual_seed(seed)

    # ============ Logger ============ #
    log_dir = HydraConfig.get().runtime.output_dir
    writer, _, new_log_dir = logger(cfg, log_dir)

    terminal_console = Console()  # Terminal output
    file_name = f"train.log"
    file_console = Console(
        file=open(file_name, "w"),
    )

    def log_print(*args, **kwargs):
        """Log to both terminal and file with immediate flushing"""
        # Print to terminal
        terminal_console.print(*args, **kwargs)
        # Print to file and flush immediately
        file_console.print(*args, **kwargs)
        file_console.file.flush()  # Force immediate write to disk

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n[bold blue]{'=' * 50}[/bold blue]")
    log_print(f"[bold blue]Starting at {timestamp}[/bold blue]")
    log_print(f"[bold green]Configuration:[/bold green]")
    log_print(OmegaConf.to_yaml(cfg))

    # ============ Load Backbones ============ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbones = nn.ModuleDict()
    patch_sizes_map = {}

    # Check if 'backbone' is a single config or a dictionary/list of configs
    # Ideally, configure cfg.backbones as a Dict of backbone configs
    if "backbones" in cfg:
        backbone_configs = cfg.backbones
    else:
        # Fallback for backward compatibility
        backbone_configs = {"default": cfg.backbone}

    log_print("[bold green]Loading Backbones...[/bold green]")
    all_patch_sizes = set()

    for name, bk_cfg in backbone_configs.items():
        bk_model = instantiate(bk_cfg)
        bk_model = bk_model.to(device)
        bk_model.eval()
        backbones[name] = bk_model

        p_size = bk_model.patch_size
        if isinstance(p_size, tuple): p_size = p_size[0]  # Handle (14,14)

        if p_size not in patch_sizes_map:
            patch_sizes_map[p_size] = []
        patch_sizes_map[p_size].append(name)
        all_patch_sizes.add(p_size)

        log_print(f"Loaded {name}: Patch Size={p_size}")

    log_print(f"[bold yellow]Using device: {device}[/bold yellow]")
    log_print(f"\n[bold cyan]Image size: {cfg.img_size}[/bold cyan]")

    # ============ Load Student Upsamplers ============ #
    anyup = instantiate(cfg.model)

    if getattr(cfg, "model_ckpt", None) is not None:
        run_dir = Path(cfg.model_ckpt).parent.parent
        if os.path.exists(run_dir / ".hydra" / "config.yaml"):
            log_print(f"Loading AnyUp from {cfg.model_ckpt}, original run dir: {run_dir}")
            old_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
            anyup = instantiate(old_cfg.model, _convert_="partial")  # or _recursive_=False if needed
        state = torch.load(cfg.model_ckpt, map_location="cpu", weights_only=False)
        state = state['anyup']
        anyup.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        log_print(f"Loaded AnyUp from {cfg.model_ckpt}")

    anyup.cuda()
    anyup.train()

    # ============ Preparing Datasets ============ #
    # We pass the list of ALL unique patch sizes to the dataloader
    first_backbone = list(backbones.values())[0]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class MultiPatchBackboneInfo:
        def __init__(self, p_sizes, config):
            self.patch_size = p_sizes
            self.config = config

    dummy_backbone_info = MultiPatchBackboneInfo(list(all_patch_sizes), {"mean": mean, "std": std})

    train_dataloader, _ = get_dataloaders(
        cfg,
        dummy_backbone_info,  # Passes list of patch sizes
        augmentation_strength=cfg.get("augmentation_strength", 1.0),
        mean=mean,
        std=std
    )

    log_print(f"[bold cyan]Train Dataset size: {len(train_dataloader.dataset)}[/bold cyan]")

    # ============ Get training criterion ================= #
    criterion = Cosine_MSE()
    epoch_start = 0

    # ============ preparing Optimizers ============ #
    all_params = []
    all_params.extend(list(anyup.parameters()))
    print("Number of parameters: ", sum(p.numel() for p in all_params))
    optimizer_anyup = instantiate(cfg.optimizer, params=all_params)

    total_batches = cfg.max_steps
    checkpoint_interval = int(total_batches * 0.2)

    # Calculate total training steps
    total_epochs = cfg.epochs

    # Loop
    for epoch in range(epoch_start, cfg.epochs):
        # Training loop
        for batch_idx, batch in enumerate(
                tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch}",
                )
        ):
            current_step = epoch * len(train_dataloader) + batch_idx
            overall_progress = (current_step / cfg.max_steps) * 100

            batch = get_batch(batch, device)

            # 1. Retrieve the patch size chosen by the collate function
            current_patch_size = batch["patch_size"]
            if isinstance(current_patch_size, torch.Tensor):
                current_patch_size = current_patch_size[0].item()

            # 2. Select a backbone compatible with this patch size
            available_backbones = patch_sizes_map[current_patch_size]
            selected_backbone_name = random.choice(available_backbones)
            backbone = backbones[selected_backbone_name]

            hr_image_batch = batch["hr_image"]
            lr_image_batch = batch.get("lr_image", hr_image_batch)
            guidance_image_batch = batch.get("guidance_image", hr_image_batch)

            with autocast(device_type="cuda", dtype=torch.bfloat16):

                with torch.no_grad():
                    hr_feats, _ = backbone(hr_image_batch)  # (N, ...)
                    lr_feats, _ = backbone(lr_image_batch)  # (M, ...)

                # Handle upsampling size (h, w)
                # If batch["upsampling_size"] is a tensor (from collate), extract int
                ups_size = batch.get("upsampling_size")
                if isinstance(ups_size, torch.Tensor):
                    ups_size = int(ups_size[0].item())
                h = w = ups_size if ups_size else hr_feats.shape[-2]

                # predict
                anyup_hr_feats = anyup(guidance_image_batch, lr_feats, (h, w))

                # needed for downsampling regularization
                anyup_output = anyup_hr_feats

                if anyup_hr_feats.shape[0] < hr_feats.shape[0]:
                    repeat = hr_feats.shape[0] // anyup_hr_feats.shape[0]
                    anyup_hr_feats = anyup_hr_feats.repeat_interleave(repeat, dim=0)

                # optional crop slicing
                if "guidance_crop" in batch:
                    # guidance_crop: (N,2) in patch units (row, col)
                    anyup_hr_feats = torch.stack([
                        anyup_hr_feats[i, :, x_i: x_i + hr_feats.shape[-2], y_i: y_i + hr_feats.shape[-1]]
                        for i, (x_i, y_i) in enumerate(batch["guidance_crop"])
                    ], dim=0)

                # ============ Loss AnyUp ============ #
                loss = {"anyup_hr": criterion(anyup_hr_feats, hr_feats)["total"], "anyup_reg": 0.0, "anyup_down": 0.0}

                if "guidance_crop" in batch:
                    feats = lr_feats
                else:
                    feats = hr_feats

                weight = getattr(cfg, "augmentation_regularization", 0.1)
                if weight > 0:
                    noised_guidance_image = batch.get("augmented_guidance_image")
                    pred_uhr_feats = anyup(noised_guidance_image, feats, (224, 224))
                    with torch.no_grad():
                        target_uhr_feats = anyup(guidance_image_batch, feats, (224, 224))

                    loss["anyup_reg"] = criterion(pred_uhr_feats, target_uhr_feats.detach())["total"] * weight
                else:
                    loss["anyup_reg"] = 0

                weight = getattr(cfg, "downsampling_regularization", 0)
                if weight > 0:
                    loss["anyup_down"] = \
                    criterion(F.interpolate(anyup_output, size=feats.shape[-2:], mode="area"), feats)["total"] * weight
                else:
                    loss["anyup_down"] = 0

            loss_anyup = loss["anyup_hr"] + loss["anyup_reg"] + loss["anyup_down"]

            optimizer_anyup.zero_grad()
            loss_anyup.backward()
            optimizer_anyup.step()

            # Optional: Update tqdm with loss information
            if batch_idx % FREQ == 0:
                # Log all loss components to tensorboard
                for loss_name, loss_value in loss.items():
                    if loss_value != 0:
                        writer.add_scalar(
                            f"Loss/{loss_name}",
                            loss_value.item(),
                            current_step,
                        )

                # Log learning rates
                writer.add_scalar(
                    "Learning Rate AnyUp",
                    optimizer_anyup.param_groups[0]["lr"],
                    current_step,
                )

                # Build concise log message
                loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss.items() if v != 0])
                log_print(
                    f"Epoch={epoch}/{total_epochs} | "
                    f"Batch={batch_idx}/{len(train_dataloader)} | "
                    f"Progress: {overall_progress:.1f}% | "
                    f"{loss_str}"
                )

            # Save checkpoint at every 10% interval
            if (current_step % checkpoint_interval == 0) or (current_step >= cfg.max_steps):
                checkpoint_path = os.path.join(new_log_dir, f"model_{current_step}steps.pth")

                save_dict = {
                    "optimizer_anyup": optimizer_anyup.state_dict(),
                    "epoch": epoch,
                    "cfg": cfg,
                    "anyup": anyup.state_dict(),
                }

                torch.save(save_dict, checkpoint_path)
                log_print(f"Saved checkpoint: {checkpoint_path}")

                if current_step >= cfg.max_steps:
                    writer.flush()
                    file_console.file.close()
                    return

            if cfg.sanity:
                break

    writer.flush()
    file_console.file.close()


if __name__ == "__main__":
    trainer()
