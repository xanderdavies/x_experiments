from typing import List
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # bring the tensors to the cpu
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # get the predictions
        logits, _ = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        if pl_module.log_images:
            trainer.logger.experiment.log({
                "examples": [wandb.Image(x, caption=f"Pred:{pred}, Label:{y}")
                             for x, pred, y in zip(val_imgs[:self.num_samples],
                                                   preds[:self.num_samples],
                                                   val_labels[:self.num_samples])]
            })


class LitModel(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, fc_activation=F.relu, top_k=None, opt_str='ADAM', log_images=False):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.fc_activation = fc_activation
        self.top_k = top_k
        self.opt_str = opt_str
        self.log_images = log_images

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.pool1 = torch.nn.MaxPool2d(2)
        self.pool2 = torch.nn.MaxPool2d(2)

        n_sizes = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_sizes, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    # returns the size of the output tensor going into Linear layer from the conv block
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        return x

    # will be used during inference
    def forward(self, x):
        intermediate_activations = []
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        # also return the activations for each layer
        x = self.fc_activation(self.fc1(x))
        if self.top_k is not None:
            x, _ = torch.topk(x, self.top_k)
        intermediate_activations.append(x)
        x = self.fc_activation(self.fc2(x))
        if self.top_k is not None:
            x, _ = torch.topk(x, self.top_k)
        intermediate_activations.append(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x, intermediate_activations

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.forward(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, prog_bar=True,
                 on_epoch=True, on_step=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, intermediate_activations = self.forward(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        self.log_dead_neurons(intermediate_activations)
        self.log_avg_number_active_neurons(intermediate_activations)

        return loss

    def log_dead_neurons(self, intermediate_activations: List[torch.Tensor]):
        fc1_activations, fc2_activations = intermediate_activations
        fc1_activations = fc1_activations.detach().cpu().numpy()
        fc2_activations = fc2_activations.detach().cpu().numpy()

        # get the dead neurons and divide total dead neurons (across batches) by the batch size * number of neurons in the layer
        fc1_activity_per_neuron = fc1_activations.sum(axis=0)
        fc2_activity_per_neuron = fc2_activations.sum(axis=0)
        fc1_dead_neurons = np.sum(fc1_activity_per_neuron <= 0.1)
        fc2_dead_neurons = np.sum(fc2_activity_per_neuron <= 0.1)
        fc1_dead_neurons_percent = fc1_dead_neurons / fc1_activations.shape[1]
        fc2_dead_neurons_percent = fc2_dead_neurons / fc1_activations.shape[1]

        # log the dead neurons
        self.logger.experiment.log({
            "fc1_dead_neuron_prevalence": fc1_dead_neurons_percent,
            "fc2_dead_neuron_prevalence": fc2_dead_neurons_percent,
        })

    def log_avg_number_active_neurons(self, intermediate_activations: List[torch.Tensor]):
        fc1_activations, fc2_activations = intermediate_activations
        fc1_activations = fc1_activations.detach().cpu().numpy()
        fc2_activations = fc2_activations.detach().cpu().numpy()

        # get the number of neurons that are active across batches
        fc1_active_neurons = np.sum(fc1_activations > 0.1, axis=0)
        fc2_active_neurons = np.sum(fc2_activations > 0.1, axis=0)

        # log the average number of active neurons
        self.logger.experiment.log({
            "fc1_avg_active_neurons": np.mean(fc1_active_neurons),
            "fc2_avg_active_neurons": np.mean(fc2_active_neurons),
        })

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.forward(x)
        loss = F.nll_loss(logits, y)

        # test metrics
        preds = torch.argmax(logits)
        acc = accuracy(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.opt_str == "ADAM":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate)
        elif self.opt_str == "SGDM":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer {self.opt_str}")
        return optimizer
