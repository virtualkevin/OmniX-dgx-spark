import os
from typing import Optional

import hydra
import torch

# torch.multiprocessing.set_sharing_strategy('file_system')

import rootutils
from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

# Setup project root for imports and env variables
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sys
ROOT = os.environ["PROJECT_ROOT"]
deformable_detr_path = os.path.join(ROOT, "dependencies", "Deformable_DETR")
if deformable_detr_path not in sys.path:
    sys.path.insert(0, deformable_detr_path)
    
# Global logger (only logs from rank 0 in multi-GPU training)
from src.utils.pylogger import RankedLogger
log = RankedLogger(__name__, rank_zero_only=True)

def train(cfg: DictConfig):
    """
    Main training loop: initialize data, model, callbacks, logger, trainer.
    Uses output_root/exp_name folder structure:
      tensorboard/   -> TensorBoard logs
      checkpoint/    -> Model checkpoints
      config.yaml    -> Final used configuration
    """
    
    # Use medium precision for matmul operations (speed optimization)
    torch.set_float32_matmul_precision("medium")

    # Set global random seed (same for all ranks)
    if cfg.get("seed") is not None:
        from lightning import seed_everything
        seed_everything(cfg.seed, workers=True)

    # Build experiment directories
    exp_dir = os.path.join(cfg.paths.output_root, cfg.paths.exp_name)
    tb_dir = os.path.join(exp_dir, "tensorboard")
    ckpt_dir = os.path.join(exp_dir, "checkpoint")

    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Instantiate DataModule from config
    log.info(f"Instantiating datamodule <{cfg.data_module._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data_module)
    
    # Instantiate Model from config
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # ModelCheckpoint callback: save every N epochs and last checkpoint
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}",            # checkpoint name pattern
        save_last=False,
        monitor="trainer/epoch",
        mode="max",
        save_top_k=1,                  # save last 3 epochs
        every_n_epochs=cfg.get("save_every_n_epochs", 20)
    )

    # TensorBoard logger: logs stored in tb_dir
    logger = TensorBoardLogger(save_dir=tb_dir, name="", version="")

    # Instantiate Trainer from config
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[checkpoint_cb],
        logger=logger,
        enable_progress_bar=True,
    )

    # Training phase
    if cfg.get("train", True):
        log.info("Starting training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Testing phase
    if cfg.get("test", False):
        log.info("Starting testing...")
        trainer.test(model=model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """
    Hydra main entry point.
    Loads config and runs the training loop.
    """
    
    train(cfg)
    return None


if __name__ == "__main__":
    main()
