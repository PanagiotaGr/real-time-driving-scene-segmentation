import os
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from typing import Dict, Any, Optional
from src.utils.metrics import SegmentationMetrics

class Trainer:
    """
    A professional trainer class for semantic segmentation tasks.
    Handles the training loop, validation, checkpointing, and early stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LearnerLRProps] = None,
        device: str = "cpu",
        config: Any = None,
        num_classes: int = 32,
        early_stopping_patience: int = 10
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

        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_model_state = None

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_pixel_acc': [],
            'val_pixel_acc': [],
            'val_mIoU': []
        }

        # Create directories
        os.makedirs(self.config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.training.log_dir, exist_ok=True)

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Always save latest
        latest_path = os.path.join(self.config.training.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.config.training.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"--> Saved best model to {best_path}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_pixels = 0
        total_correct = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs} [Train]")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy for logging
            preds = torch.argmax(outputs, dim=1)
            correct = (preds == masks).sum().item()
            total = masks.numel()

            running_acc += correct / total
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.train_loader)
        avg_acc = running_acc / len(self.train_loader)

        return {'loss': avg_loss, 'acc': avg_acc}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0

        # For mIoU
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs} [Val]")

        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            # Accuracy
            correct = (preds == masks).sum().item()
            total = masks.numel()
            running_acc += correct / total

            # Accumulate for IoU
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = running_loss / len(self.val_loader)
        avg_acc = running_acc / len(self.val_loader)

        # Calculate mIoU
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        iou_results = self.metrics.iou_metrics(all_preds, all_targets)
        mIoU = iou_results['mIoU']

        return {
            'loss': avg_loss,
            'acc': avg_acc,
            'mIoU': mIoU
        }

    def fit(self):
        print(f"Starting training on {self.device}...")

        for epoch in range(self.config.training.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Log History
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_pixel_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_pixel_acc'].append(val_metrics['acc'])
            self.history['val_mIoU'].append(val_metrics['mIoU'])

            print(f"\nSummary Epoch {epoch+1}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | mIoU: {val_metrics['mIoU']:.4f}")

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Checkpoint & Early Stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.counter += 1
                if self.counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

            print("-" * 30)

        print("Training complete.")
        return self.history
