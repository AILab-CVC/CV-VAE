import torchdata.datapipes as dp
import json
from PIL import Image
import functools
import numpy as np
import torch
import pickle
import os
import random
from torchvision import transforms
from braceexpand import braceexpand
from omegaconf import ListConfig, DictConfig

# import hydra
import tarfile
import importlib
from typing import (
    cast,
    IO,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Dict,
    Union,
    Callable,
    List,
    Any,
)
import torch.distributed as dist
from io import BufferedIOBase
from torchdata.datapipes.utils import StreamWrapper
import warnings
from torchdata.datapipes.iter import IterDataPipe
import webdataset as wds
from torch.utils.data.datapipes.iter.sharding import (
    SHARDING_PRIORITIES,
    ShardingFilterIterDataPipe,
)
from torchdata.dataloader2 import (
    DataLoader2,
    MultiProcessingReadingService,
    DistributedReadingService,
    SequentialReadingService,
)

from ..util import instantiate_from_config


def create_obj(string: str, reload: bool = False, invalidate_cache: bool = True) -> Any:
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def make_callable(config):
    return functools.partial(
        create_obj(config["target"]), **config.get("params", dict())
    )


def make_callable_with_children(config):
    params = config.get("params", dict())
    new_params = dict(config.get("params", dict()))
    for key, value in params.items():
        if isinstance(value, (dict, DictConfig)) and "target" in value:
            new_params[key] = instantiate_from_config(value)
        else:
            new_params[key] = value

    return functools.partial(create_obj(config["target"]), **new_params)


def apply_sharding(datapipe):
    if dist.is_available() and dist.is_initialized():
        # after this operation datapipes in the distinct processes contain different tars

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        datapipe.apply_sharding(world_size, global_rank)
        print("#" * 100)
        print(f"distributing shards for worker with global rank {global_rank}")
        print("#" * 100)

    else:
        print(f"torch distributed not used, not applying sharding.")

    return datapipe


def reorg_wds_sample(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    # if 'metadata' not in unwarpped:
    #     unwarpped['metadata'] = '{}'
    # if '__key__' in unwarpped:
    #     unwarpped['__key__'] = unwarpped['__key__'].split('/')[-1]
    return unwarpped


def dict_collation_fn(
    samples: List, combine_tensors: bool = True, combine_scalars: bool = True
) -> Dict:
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])

    del samples
    del batched
    return result


def dict_collation_fn_with_concat(
    samples: List,
    concat_keys: Optional[List[str]] = None,
    rank0_keys: Optional[List[str]] = None,
) -> Dict:
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if rank0_keys is not None and key in rank0_keys:
                result[key] = list(batched[key])[0]
            else:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if concat_keys is not None and key in concat_keys:
                result[key] = torch.cat(list(batched[key]), dim=0)
            else:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])

    del samples
    del batched
    return result


def create_single_dataset(
    urls_or_dir: Optional[Union[List[str], str]] = None,
    meta_urls_or_dir: Optional[Union[List[str], str]] = None,
    file_mask: str = "*.tar",
    repeat: int = None,
    shardshuffle: int = 10000,
    sample_shuffle: int = 1,
    batch_size: int = None,
    collation_fn: Optional[Union[Callable, Dict, DictConfig]] = None,
    handler: Union[Callable, DictConfig] = wds.reraise_exception,
    decoder: Union[Callable, DictConfig] = None,
    filter: Union[Callable, DictConfig] = None,
    inputs_selector: Union[Callable, DictConfig] = None,
):

    assert (
        int(urls_or_dir is not None) + int(meta_urls_or_dir is not None) == 1
    ), f"One of the (urls_or_dir, meta_urls_or_dir) must be None."

    assert decoder is not None, f"decoder must be bot None."

    assert file_mask in ["*.tar", "*.csv", "*.jsonl"]

    is_wds = urls_or_dir is not None
    urls_or_dir = urls_or_dir or meta_urls_or_dir

    if isinstance(urls_or_dir, (List, ListConfig, list)):
        urls = list(urls_or_dir)
    elif isinstance(urls_or_dir, str):
        urls = list(braceexpand(urls_or_dir))
    else:
        raise TypeError(
            "urls need to be path to a S3 prefix or dir or list of paths to more than one prefixes"
        )

    if isinstance(handler, (DictConfig, Dict)):
        handler = make_callable(handler)

    if isinstance(decoder, (DictConfig, Dict)):
        decoder = make_callable_with_children(decoder)

    if isinstance(filter, (DictConfig, Dict)):
        filter = make_callable(filter)

    if isinstance(inputs_selector, (DictConfig, Dict)):
        inputs_selector = make_callable(inputs_selector)

    if not collation_fn:
        collation_fn = dict_collation_fn

    if isinstance(collation_fn, (Dict, DictConfig)):
        collation_fn = make_callable(collation_fn)

    datapipe = dp.iter.FileLister(root=urls, masks=file_mask, recursive=True)
    datapipe = datapipe.cycle(count=repeat)
    datapipe = datapipe.shuffle(buffer_size=shardshuffle)
    if is_wds:
        datapipe = datapipe.sharding_filter()
        # datapipe = apply_sharding(datapipe)

        datapipe = datapipe.open_files(mode="b")
        datapipe = datapipe.load_from_tar_with_handler(handler=handler)
        datapipe = datapipe.map(decoder)
        datapipe = datapipe.webdataset()
        datapipe = datapipe.map(reorg_wds_sample)
        if filter is not None:
            datapipe = datapipe.filter(filter)

        if inputs_selector is not None:
            datapipe = datapipe.map(inputs_selector)

        if sample_shuffle > 1:
            datapipe = datapipe.shuffle(buffer_size=sample_shuffle)

    else:
        # dataset is consist of meta_info (in meta_urls_or_dir) and data (in data_root)
        datapipe = datapipe.open_files(mode="r")
        if file_mask == "*.csv":
            datapipe = datapipe.parse_csv_as_dict()
        elif file_mask == "*.jsonl":
            datapipe = datapipe.parse_jsonl_files_with_handler(handler=handler)
        else:
            raise NotImplementedError(f"file_type: {file_mask} is not support now!")

        datapipe = datapipe.sharding_filter()
        datapipe = apply_sharding(datapipe)
        if sample_shuffle > 1:
            datapipe = datapipe.shuffle(buffer_size=sample_shuffle)
        datapipe = datapipe.map(decoder)
        if filter is not None:
            datapipe = datapipe.filter(filter)

        if inputs_selector is not None:
            datapipe = datapipe.map(inputs_selector)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(collate_fn=collation_fn)

    return datapipe


def create_multi_dataset(
    datasets: Optional[ListConfig] = None,
    sample_weights: Optional[list] = None,
    shardshuffle: int = 10000,
    sample_shuffle: int = 1,
    batch_size: int = None,
    collation_fn: Optional[Union[Callable, Dict, DictConfig]] = None,
    handler: Union[Callable, DictConfig] = wds.reraise_exception,
    decoder: Union[Callable, DictConfig] = None,
    filter: Union[Callable, DictConfig] = None,
    inputs_selector: Union[Callable, DictConfig] = None,
    seed: int = 58,
):
    assert len(datasets) >= 1, "Num. of datasets must >= 1."
    if sample_weights is None:
        sample_weights = [1.0] * len(datasets)

    assert len(sample_weights) == len(
        datasets
    ), f"len(sample_weights) = {len(sample_weights)}, len(datasets) = {len(datasets)}"

    datapipes = []
    for name, datapipe_config in datasets.items():
        datapipe_config["shardshuffle"] = datapipe_config.get(
            "shardshuffle", shardshuffle
        )
        datapipe_config["sample_shuffle"] = datapipe_config.get(
            "sample_shuffle", sample_shuffle
        )
        datapipe_config["batch_size"] = datapipe_config.get("batch_size", batch_size)
        datapipe_config["collation_fn"] = datapipe_config.get(
            "collation_fn", collation_fn
        )
        datapipe_config["handler"] = datapipe_config.get("handler", handler)
        datapipe_config["decoder"] = datapipe_config.get("decoder", decoder)
        datapipe_config["filter"] = datapipe_config.get("filter", filter)
        datapipe_config["inputs_selector"] = datapipe_config.get(
            "inputs_selector", inputs_selector
        )
        print(f"Build {name} dataset with config: {datapipe_config}")
        datapipe = create_single_dataset(**datapipe_config)
        datapipes.append(datapipe)

    datasets_and_weights = {}
    for dataset, sample_weight in zip(datapipes, sample_weights):
        datasets_and_weights[dataset] = sample_weight

    if dist.is_available() and dist.is_initialized():
        seed = seed + dist.get_rank()

    datapipe = dp.iter.SampleMultiplexer(datasets_and_weights, seed=seed)

    return datapipe


def create_dataloader(
    datapipe: IterDataPipe,
    batch_size: Optional[int] = None,
    collation_fn: Optional[Union[Callable, Dict, DictConfig]] = None,
    num_workers: int = 1,
    dataloader_kwargs: Optional[Union[Dict, DictConfig]] = None,
) -> torch.utils.data.DataLoader:

    if not dataloader_kwargs:
        dataloader_kwargs = {}

    if not collation_fn:
        collation_fn = dict_collation_fn

    if isinstance(collation_fn, (Dict, DictConfig)):
        collation_fn = make_callable(collation_fn)

    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate(collate_fn=collation_fn)

    # create loader
    dataloader = torch.utils.data.DataLoader(
        datapipe, batch_size=None, num_workers=num_workers, **dataloader_kwargs
    )
    # mp_service = MultiProcessingReadingService(num_workers=num_workers)
    # dist_service = DistributedReadingService()
    # reading_service = SequentialReadingService(dist_service, mp_service)
    # dataloader = DataLoader2(datapipe, reading_service=reading_service)
    return dataloader
