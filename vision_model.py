import os
import torch
from torchmetrics import Accuracy
import wandb
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
import torchvision.models as models
from typing import Optional
import torchvision.datasets as datasets
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torchvision import transforms
from pytorch_lightning.utilities.cli import LightningCLI


class ImageNetLightningModule(pl.LightningModule):
    def __init__(
        self,
        data_path: str,
        arch: str = "resnet18",
        pretrained: bool = False,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        workers: int = 4,
    ):
        super().__init__()
        self.data_path = data_path
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.workers = workers
        self.model = models.__dict__[self.arch](pretrained=self.pretrained)
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.train_acc1 = Accuracy(topk=1)
        self.train_acc5 = Accuracy(topk=5)
        self.eval_acc1 = Accuracy(topk=1)
        self.eval_acc5 = Accuracy(topk=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.forward(images)
        train_loss = F.cross_entropy(output, targets)
        self.log("train_loss", train_loss)
        # log accuracy
        self.train_acc1(output, targets)
        self.train_acc5(output, targets)
        self.log("train_acc1", self.train_acc1, prog_bar=True)
        self.log("train_acc5", self.train_acc5, prog_bar=True)
        return train_loss

    def eval_step(self, batch, batch_idx, prefix):
        images, targets = batch
        output = self.forward(images)
        eval_loss = F.cross_entropy(output, targets)
        self.log(f"{prefix}_loss", eval_loss)
        # log accuracy
        self.eval_acc1(output, targets)
        self.eval_acc5(output, targets)
        self.log(f"{prefix}_acc1", self.eval_acc1, prog_bar=True)
        self.log(f"{prefix}_acc5", self.eval_acc5, prog_bar=True)
        return eval_loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def training_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        # lr is decayed by a factor of 0.1 every 30 epochs
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.1 ** (epoch // 30))
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            train_dir = os.path.join(self.data_path, "train")
            self.train_dataset = datasets.ImageFolder(
                train_dir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
         # all stages will use the eval dataset
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        val_dir = os.path.join(self.data_path, "val")
        self.eval_dataset = datasets.ImageFolder(
            val_dir,
            transforms.Compose([transforms.Resize(256), transforms.CenterCrop(
                224), transforms.ToTensor(), normalize]),
        )


if __name__ == "__main__":
    LightningCLI(
        ImageNetLightningModule,
        trainer_defaults={
            "max_epochs": 90,
            "accelerator": "auto",
            "devices": 1,
            "logger": False,
            "benchmark": True,
            "callbacks": [
                # the PyTorch example refreshes every 10 batches
                TQDMProgressBar(refresh_rate=10),
                # save when the validation top1 accuracy improves
                ModelCheckpoint(monitor="val_acc1", mode="max"),
            ],
        },
        seed_everything_default=42,
        save_config_overwrite=True,
    )
