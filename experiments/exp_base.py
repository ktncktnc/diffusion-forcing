"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Literal, List, Dict
import pathlib
import os

import hydra
import torch
from lightning.pytorch.strategies.ddp import DDPStrategy

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from omegaconf import DictConfig

from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero
torch.set_float32_matmul_precision("high")


class BaseExperiment(ABC):
    """
    Abstract class for an experiment. This generalizes the pytorch lightning Trainer & lightning Module to more
    flexible experiments that doesn't fit in the typical ml loop, e.g. multi-stage reinforcement learning benchmarks.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError
    need_original_algo: List[str] = []

    def __init__(
        self,
        root_cfg: DictConfig,
        logger: Optional[WandbLogger] = None,
        ckpt_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Constructor

        Args:
            cfg: configuration file that contains everything about the experiment
            logger: a pytorch-lightning WandbLogger instance
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug
        self.logger = logger
        self.ckpt_path = ckpt_path
        self.algo = None

    def _build_algo(self):
        """
        Build the lightning module
        :return:  a pytorch-lightning module to be launched
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        
        # Load original algo
        if algo_name in self.need_original_algo:
            original_algo_name = self.root_cfg.algorithm.original_algo._name
            if original_algo_name not in self.compatible_algorithms:
                raise ValueError(
                    f"Original Algorithm {original_algo_name} not found in compatible_algorithms for this Experiment class. "
                    "Make sure you define compatible_algorithms correctly and make sure that each key has "
                    "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
                )
            
            # If has ckpt
            if self.root_cfg.algorithm.original_algo.ckpt_path:
                print(f"Loading original algo from {self.root_cfg.algorithm.original_algo.ckpt_path}")
                original_algo = self.compatible_algorithms[original_algo_name].load_from_checkpoint(
                    self.root_cfg.algorithm.original_algo.ckpt_path,
                    cfg=self.root_cfg.algorithm.original_algo    
                )
            else:
                print(f"Building original algo from scratch")
                original_algo = self.compatible_algorithms[original_algo_name](
                    cfg=self.root_cfg.algorithm.original_algo
                )

            return self.compatible_algorithms[algo_name](
                original_algo=original_algo,
                cfg=self.root_cfg.algorithm
            )

        return self.compatible_algorithms[algo_name](self.root_cfg.algorithm)

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        In most computer vision / nlp applications, tasks should be just train and test.
        In reinforcement learning, you might have more stages such as collecting dataset etc

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )


class BaseLightningExperiment(BaseExperiment):
    """
    Abstract class for pytorch lightning experiments. Useful for computer vision & nlp where main components are
    simply models, datasets and train loop.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    # each key has to be a yaml file under '[project_root]/configurations/dataset' without .yaml suffix
    compatible_datasets: Dict = NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only build the config mapping for error datasets

    def _build_trainer_callbacks(self):
        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))

    def _build_training_loader(self, **kwargs) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        """
        Build the training data loader
        If kwargs are passed, they will be passed to the dataset constructor
        :return: a pytorch dataloader

        """
        train_dataset = self._build_dataset("training", dataset_cfg=kwargs.get("dataset_cfg", None))
        shuffle = (
            False if isinstance(train_dataset, torch.utils.data.IterableDataset) else self.cfg.training.data.shuffle
        )
        shuffle = kwargs.get("shuffle", shuffle)
        batch_size = kwargs.get("batch_size", self.cfg.training.batch_size)

        if train_dataset:
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=min(os.cpu_count(), self.cfg.training.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_validation_loader(self, **kwargs) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        validation_dataset = self._build_dataset("validation", dataset_cfg=kwargs.get("dataset_cfg", None))
        shuffle = (
            False
            if isinstance(validation_dataset, torch.utils.data.IterableDataset)
            else self.cfg.validation.data.shuffle
        )
        shuffle = kwargs.get("shuffle", shuffle)
        batch_size = kwargs.get("batch_size", self.cfg.validation.batch_size)
        if validation_dataset:
            return torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=batch_size,
                num_workers=min(os.cpu_count(), self.cfg.validation.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_test_loader(self, **kwargs) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        test_dataset = self._build_dataset("test", dataset_cfg=kwargs.get("dataset_cfg", None))
        shuffle = False if isinstance(test_dataset, torch.utils.data.IterableDataset) else self.cfg.test.data.shuffle
        shuffle = kwargs.get("shuffle", shuffle)
        batch_size = kwargs.get("batch_size", self.cfg.test.batch_size)

        if test_dataset:
            return torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=min(os.cpu_count(), self.cfg.test.data.num_workers),
                shuffle=shuffle,
                persistent_workers=True,
            )
        else:
            return None

    def _build_loader(self, split: str, **kwargs) -> Optional[Union[TRAIN_DATALOADERS, pl.LightningDataModule]]:
        if split == "training":
            return self._build_training_loader(**kwargs)
        elif split == "validation":
            return self._build_validation_loader(**kwargs)
        elif split == "test":
            return self._build_test_loader(**kwargs)
        else:
            raise ValueError(f"split '{split}' not implemented")
        

    def training(self) -> None:
        """
        All training happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.training.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []
        if self.logger:
            callbacks.append(LearningRateMonitor("step", True))
        if "checkpointing" in self.cfg.training:
            callbacks.append(
                ModelCheckpoint(
                    pathlib.Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]) / "checkpoints",
                    **self.cfg.training.checkpointing,
                )
            )
        callbacks.append(TQDMProgressBar(refresh_rate=100))

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger if self.logger else False,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            gradient_clip_val=self.cfg.training.optim.gradient_clip_val,
            val_check_interval=self.cfg.validation.val_every_n_step,
            limit_val_batches=self.cfg.validation.limit_batch,
            check_val_every_n_epoch=self.cfg.validation.val_every_n_epoch,
            accumulate_grad_batches=self.cfg.training.optim.accumulate_grad_batches,
            precision=self.cfg.training.precision,
            detect_anomaly=False,  # self.cfg.debug,
            num_sanity_val_steps=int(self.cfg.debug),
            max_epochs=self.cfg.training.max_epochs,
            max_steps=self.cfg.training.max_steps,
            max_time=self.cfg.training.max_time,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.fit(
            self.algo,
            train_dataloaders=self._build_training_loader(),
            val_dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def validation(self) -> None:
        """
        All validation happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.validation.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_val_batches=self.cfg.validation.limit_batch,
            precision=self.cfg.validation.precision,
            detect_anomaly=False,  # self.cfg.debug,
            inference_mode=self.cfg.validation.inference_mode,
        )

        # if self.debug:
        #     self.logger.watch(self.algo, log="all")

        trainer.strategy.strict_loading = False
        trainer.validate(
            self.algo,
            dataloaders=self._build_validation_loader(),
            ckpt_path=self.ckpt_path,
        )

    def test(self) -> None:
        """
        All testing happens here
        """
        if not self.algo:
            self.algo = self._build_algo()
        if self.cfg.test.compile:
            self.algo = torch.compile(self.algo)

        callbacks = []

        trainer = pl.Trainer(
            accelerator="auto",
            logger=self.logger,
            devices="auto",
            num_nodes=self.cfg.num_nodes,
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            callbacks=callbacks,
            limit_test_batches=self.cfg.test.limit_batch,
            precision=self.cfg.test.precision,
            detect_anomaly=False,  # self.cfg.debug,
        )

        # Only load the checkpoint if only testing. Otherwise, it will have been loaded
        # and further trained during train.
        trainer.test(
            self.algo,
            dataloaders=self._build_test_loader(),
            ckpt_path=self.ckpt_path,
        )

    def _build_dataset(self, split: str, dataset_cfg: DictConfig = None) -> Optional[torch.utils.data.Dataset]:
        dataset_cfg = dataset_cfg if dataset_cfg else self.root_cfg.dataset

        kwargs = {}
        # if 'error' in dataset_cfg._name:
        #     assert self.algo is not None, "Algorithm must be built before building prediction dataset"
        #     kwargs['model'] = self.algo
        #     kwargs['original_dataloader'] = self._build_loader(split, dataset_cfg=dataset_cfg.original_dataset, shuffle=False)
        #     kwargs['root_dir'] = self.config_mapping.get_value(self.algo.original_algo)

        if split in ["training", "test", "validation"]:
            # TODO: get root_dir from original algo
            return self.compatible_datasets[dataset_cfg._name](dataset_cfg, split=split, **kwargs)
        else:
            raise NotImplementedError(f"split '{split}' is not implemented")

    # def _build_config_mapping(self):
    #     """
    #     Build the configuration mapping for the experiment. Only for prediction datasets
    #     """
    #     self.config_mapping = ConfigMapping.load_from_json(self.cfg.config_mapping_path)