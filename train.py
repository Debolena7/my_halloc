import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src.module import get_module
from src.datamodule import get_datamodule
from src.utils.utils import get_pylogger
from src.message import BaseTrainerMessage

import torch
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

def get_callbacks(callback_cfg: DictConfig) -> list:
    _callbacks = []
    for k, v in callback_cfg.items():
        _callbacks.append(eval(f"callbacks.{k}")(**v))
    return _callbacks


def get_trainer_msg(trainer: Trainer) -> BaseTrainerMessage:
    return BaseTrainerMessage(
        max_epochs=trainer.max_epochs,
        num_nodes=trainer.num_nodes,
        num_devices=trainer.num_devices,
    )


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def train(cfg: DictConfig) -> None:
    logger = get_pylogger(__name__)
    
    logger.info(OmegaConf.to_yaml(cfg))

    # Fix seed via Pytorch Lightning.
    # Set trainer.deterministic as True for reproducibility.
    seed = cfg.experiment.get("seed")
    if seed is not None:
        seed_everything(seed, workers=True)

    logger.info("Instantiating callbacks...")
    callbacks = get_callbacks(cfg.callback)

    logger.info("Instantiating tensorboard logger...")
    tb_logger = TensorBoardLogger(
        save_dir=cfg.experiment.work_dir,
        name=cfg.experiment.name,
        sub_dir="tensorboard",
        default_hp_metric=False,
    )

    logger.info("Instantiating trainer...")

    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        **cfg.trainer,
    )

    trainer_msg = get_trainer_msg(trainer=trainer)

    logger.info(f"Instantiating datamodule: {cfg.datamodule.name}")
    datamodule = get_datamodule(cfg.datamodule.name)(
        **cfg.datamodule.params,
        trainer_msg=trainer_msg,
        train_dataset_cfg=cfg.train_dataset,
        valid_dataset_cfg=cfg.valid_dataset,
    )
    datamodule.setup(stage="fit")
    datamodule_msg = datamodule.get_datamodule_msg()

    logger.info(f"Instantiating module: {cfg.module.name}")
    module = get_module(cfg.module.name)(
        train_loss_cfg=cfg.train_loss,
        valid_loss_cfg=cfg.valid_loss,
        metric_cfg=cfg.metric,
        model_cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        datamodule_msg=datamodule_msg,
    )

    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    train()
