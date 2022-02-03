import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
from torch.utils.data import DataLoader


@enumerate
class DataSet:
    CIFAR10 = 0


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers: int = 1, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self) -> None:
        # download if needed
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        # Assign train/val datasets for use in dataloaders
        cifar_full = CIFAR10(self.data_dir, train=True,
                             transform=self.transform)
        self.cifar_train, self.cifar_val = random_split(
            cifar_full, [45000, 5000])

        self.cifar_test = CIFAR10(
            self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
