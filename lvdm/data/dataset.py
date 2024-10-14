from typing import Optional

import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from .datapipe_utils import (
    create_multi_dataset,
    create_single_dataset,
    create_dataloader,
)

try:
    from sdata import create_dataset, create_dummy_dataset, create_loader
except ImportError as e:
    # print("#" * 100)
    # print("Datasets not yet available")
    # print("to enable, we need to add stable-datasets as a submodule")
    # print("please use ``git submodule update --init --recursive``")
    # print("and do ``pip install -e stable-datasets/`` from the root of this repo")
    # print("#" * 100)
    print("#" * 100)
    print("stable-datasets not yet available")
    print("to enable, we need to add stable-datasets as a submodule")
    print("please use ``git submodule update --init --recursive``")
    print("and do ``pip install -e stable-datasets/`` from the root of this repo")
    print("Ignore the info when using CustomDataModuleFromConfig.")
    print("#" * 100)


class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.train_config = train
        assert (
            "datapipeline" in self.train_config and "loader" in self.train_config
        ), "train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
            ), "test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("USING DUMMY DATASET: HOPE YOU'RE DEBUGGING ;)")
            print("#" * 100)

    def setup(self, stage: str) -> None:
        print("Preparing datasets")
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset

        self.train_datapipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
        loader = create_loader(self.train_datapipeline, **self.train_config.loader)
        return loader

    def val_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.val_datapipeline, **self.val_config.loader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_loader(self.test_datapipeline, **self.test_config.loader)


class CustomDataModuleFromConfig(LightningDataModule):
    def __init__(
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
    ):
        super().__init__()
        self.train_config = train

        assert (
            int("datapipe" in self.train_config) + int("datapipes" in self.train_config)
            == 1
        ), "train config requires the fields `datapipe` or `datapipes`"
        assert (
            "dataloader" in self.train_config
        ), "train config requires the fields `dataloader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
                assert (
                    int("datapipe" in self.val_config)
                    + int("datapipes" in self.val_config)
                    == 1
                ), "validation config requires the fields `datapipe` or `datapipes`"
                assert (
                    "dataloader" in self.val_config
                ), "validation config requires the fields `dataloader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                int("datapipe" in self.test_config)
                + int("datapipes" in self.test_config)
                == 1
            ), "test config requires the fields `datapipe` or `datapipes`"
            assert (
                "dataloader" in self.test_config
            ), "test config requires the fields `dataloader`"

    def setup(self, stage: str) -> None:
        print("Preparing datasets")

        if "datapipes" in self.train_config:
            self.train_datapipe = create_multi_dataset(**self.train_config.datapipes)
        else:
            self.train_datapipe = create_single_dataset(**self.train_config.datapipe)

        if self.val_config:
            if "datapipes" in self.val_config:
                self.val_datapipe = create_multi_dataset(**self.val_config.datapipes)
            else:
                self.val_datapipe = create_single_dataset(**self.val_config.datapipe)
        if self.test_config:
            if "datapipes" in self.test_config:
                self.test_datapipe = create_multi_dataset(**self.test_config.datapipes)
            else:
                self.test_datapipe = create_single_dataset(**self.test_config.datapipe)

    def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
        dataloader = create_dataloader(
            self.train_datapipe, **self.train_config.dataloader
        )
        return dataloader

    def val_dataloader(self) -> wds.DataPipeline:
        return create_dataloader(self.val_datapipe, **self.val_config.dataloader)

    def test_dataloader(self) -> wds.DataPipeline:
        return create_dataloader(self.test_datapipe, **self.test_config.dataloader)
