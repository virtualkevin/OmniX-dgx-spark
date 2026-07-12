# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader
from src.data.datasets import build_dataset, get_data_loader
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy


class CustomDataModule(LightningDataModule):
    """LightningDataModule for the custom dataset.

     A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    """
    def __init__(
        self,
        train_datasets: list[str],
        validation_datasets: list[str],
        images_per_gpu: int = 24,
        num_workers: int = 12,
        num_workers_val: int = 2,
        pin_memory: bool = True,
        aspect_ratio_range: list[float] | None = None,
    ) -> None:
        """Initialize a CustomDataModule.

        :param train_dataset: Path to the training dataset.
        :param test_dataset: Path to the testing dataset.
        :param batch_size: Batch size for training and evaluation.
        :param num_workers: Number of workers for data loading.
        :param pin_memory: Whether to pin memory.
        """
        super().__init__()

        self.train_datasets = train_datasets
        self.validation_datasets = validation_datasets
        self.images_per_gpu = images_per_gpu
        self.num_workers = num_workers
        self.num_workers_val = num_workers_val
        self.pin_memory = pin_memory
        self.aspect_ratio_range = aspect_ratio_range

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

    def prepare_data(self) -> None:
        """Download or prepare the dataset if needed."""
        # Implement any dataset preparation steps if needed.
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data and set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`.
        """
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # Assert every dataset is a dict

        train_datasets = build_dataset(self.hparams.train_datasets, stage="train")
     
        print("Building train Data loader for dataset: ", self.hparams.train_datasets)
        self.train_loader = get_data_loader(
            train_datasets,
            images_per_gpu=self.images_per_gpu,
            num_workers=self.num_workers,
            pin_mem=self.pin_memory,
            shuffle=True,
            drop_last=True,
            multiprocessing_context="spawn" if isinstance(self.trainer.strategy, DeepSpeedStrategy) else None,   # for DeepSpeed ZeRO-2, for some reason the default fork context doesn't work - it would cause a "cannot allocate memory" error 
            persistent_workers=True if self.num_workers > 0 else False,
            aspect_ratio_range=self.aspect_ratio_range,
            stage="train",
        )

        # Set epoch for train and validation loaders (if applicable)
        if hasattr(self.train_loader, "dataset") and hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(0)
        if hasattr(self.train_loader, "batch_sampler") and hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(0)

        return self.train_loader

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        # construct validation datasets
        val_datasets = build_dataset(self.hparams.validation_datasets, stage="val")

        # Create individual validation data loaders for each dataset
        val_loaders = []
        for dataset in val_datasets:

            val_loaders.append(get_data_loader(
                    dataset,
                    num_workers=self.num_workers_val,
                    pin_mem=self.pin_memory,
                    shuffle=False,
                    drop_last=False,  # set to False if you want to keep the last batch, e.g., for precise evaluation
                    multiprocessing_context="spawn" if isinstance(self.trainer.strategy, DeepSpeedStrategy) else None, # for DeepSpeed ZeRO-2, for some reason the default fork context doesn't work - it would cause a "cannot allocate memory" error
                    persistent_workers=True if self.num_workers_val > 0 else False,
                    stage="val",
                ))
            
        for loader in val_loaders:
            # Set epoch for each validation loader (if applicable)
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                # print the dataset name and length
                print(f"Dataset: {loader.dataset} | Length: {len(loader.dataset)}")
                loader.dataset.set_epoch(0)
            if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
                loader.batch_sampler.set_epoch(0)

        print("Building validation CombinedLoader for datasets: ", self.hparams.validation_datasets)
        self.val_loader = CombinedLoader(val_loaders, mode='sequential')
        return self.val_loader

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Generate and save the datamodule state.

        :return: A dictionary containing the datamodule state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Reload datamodule state given datamodule `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
