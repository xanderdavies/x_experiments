# Running multiple experiments, have to be on different GPUs each. Used for SLURM.
from data_modules import DataSet
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import argparse
import torch.nn.functional as F
from wandb_model import ImagePredictionLogger, LitModel

########### EXPERIMENTS TO RUN ##############

epochs = 1000
dataset: DataSet = DataSet.CIFAR10

exp_list = [
    {
        'top_k': None,
        'fc_activation': F.relu,
        'epochs': epochs,
        'dataset': dataset,
        "opt": "ADAM",
    },
    {
        'top_k': None,
        'fc_activation': F.relu,
        'epochs': epochs,
        'dataset': dataset,
        "opt": "SGDM",
    },
    {
        'top_k': 30,
        'fc_activation': F.relu,
        'epochs': epochs,
        'dataset': dataset,
        "opt": "ADAM",
    },
    {
        'top_k': 30,
        'fc_activation': F.relu,
        'epochs': epochs,
        'dataset': dataset,
        "opt": "SGDM",
    },
]

args = None  # makes args into a global variable.


def train_func():
    exp = exp_list[args.exp_ind]

    dm = None
    if exp['dataset'] == DataSet.CIFAR10:
        dm = CIFAR10DataModule(batch_size=128, num_workers=1)
    else:
        raise NotImplementedError
    dm.prepare_data()
    dm.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image prediction
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape

    # init model
    model = LitModel(dm.size(), dm.num_classes, top_k=exp["top_k"],
                     fc_activation=exp["fc_activation"],
                     opt_str=exp["opt"])
    tags = [str(exp['dataset']), exp['opt'],
            exp['fc_activation'].__name__, exp['top_k']]
    wandb_logger = WandbLogger(
        project="cifar10_cnn", job_type='train', tags=tags)

    # init trainer
    trainer = pl.Trainer(max_epochs=exp["epochs"],
                         progress_bar_refresh_rate=20,
                         logger=wandb_logger,
                         callbacks=[ImagePredictionLogger(val_samples)],
                         checkpoint_callback=ModelCheckpoint(),
                         gpus=-1,
                         auto_select_gpus=True,
                         )

    # Train!
    trainer.fit(model, dm)

    # Close wandb logger
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_ind", type=int,
                        required=True, help="The job index.")
    parser.add_argument(
        "--num_workers", type=int, required=True, help="Num CPUS for the workers."
    )
    parser.add_argument(
        "--total_tasks",
        type=int,
        required=True,
        help="Can check the number of tasks equals the number of experiments.",
    )

    args = parser.parse_args()

    train_func()
