import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.metrics import SegmentationMetrics


class Trainer:
    """
    Trainer for semantic segmentation experiments.

    Handles training, validation, checkpointing, early stopping, metric history,
    and TensorBoard logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        device: str = "cpu",
        config: Any = None,
        num_classes: int = 32,
        early_stopping_patience: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.num_classes = num_classes
        self.early_stopping_patience = early_stopping_patience

        self.metrics = SegmentationMetrics(num_classes=num_classes)

        self.best_val_loss = float("inf")
        self.counter = 0

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_pixel_acc": [],
            "val_pixel_acc": [],
            "val_mIoU": [],
        }

        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.training.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.config.training.log_dir)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        latest_path = os.path.join(self.config.training.checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.config.training.checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"--> Saved best model to {best_path}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.training.epochs} [Train]")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == masks).sum().item()
            total = masks.numel()
            running_acc += correct / total

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.train_loader)
        avg_acc = running_acc / len(self.train_loader)

        return {"loss": avg_loss, "acc": avg_acc}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.config.training.epochs} [Val]")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == masks).sum().item()
            total = masks.numel()
            running_acc += correct / total

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.val_loader)
        avg_acc = running_acc / len(self.val_loader)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        iou_results = self.metrics.iou_metrics(all_preds, all_targets)
        mean_iou = iou_results["mIoU"]

        return {"loss": avg_loss, "acc": avg_acc, "mIoU": mean_iou}

    def _log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        self.writer.add_scalar("PixelAccuracy/train", train_metrics["acc"], epoch)
        self.writer.add_scalar("PixelAccuracy/val", val_metrics["acc"], epoch)
        self.writer.add_scalar("mIoU/val", val_metrics["mIoU"], epoch)

        current_lr = self.optimizer.param_groups[0].get("lr")
        if current_lr is not None:
            self.writer.add_scalar("LearningRate", current_lr, epoch)

    def fit(self):
        print(f"Starting training on {self.device}...")

        try:
            for epoch in range(self.config.training.epochs):
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate(epoch)

                self.history["train_loss"].append(train_metrics["loss"])
                self.history["train_pixel_acc"].append(train_metrics["acc"])
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_pixel_acc"].append(val_metrics["acc"])
                self.history["val_mIoU"].append(val_metrics["mIoU"])

                self._log_epoch(epoch, train_metrics, val_metrics)

                print(f"\nSummary Epoch {epoch + 1}:")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f}")
                print(f"  Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | mIoU: {val_metrics['mIoU']:.4f}")

                if self.scheduler:
                    self.scheduler.step()

                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.counter = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.counter += 1
                    if self.counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break

                print("-" * 30)
        finally:
            self.writer.flush()
            self.writer.close()

        print("Training complete.")
        return self.history
