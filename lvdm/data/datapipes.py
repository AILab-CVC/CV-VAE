import os
import tarfile
import time
import warnings
from io import BufferedIOBase, RawIOBase
import re
from typing import (
    Iterator,
    List,
    Tuple,
    Optional,
    cast,
    IO,
    Callable,
    Dict,
    Union,
)
import random
import gc
from collections import deque

import numpy as np
import json
import torch
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter import TarArchiveLoader
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
from torch.utils.data.datapipes.utils.common import StreamWrapper
import webdataset as wds
from typing import cast, IO, Iterable, Iterator, Optional, Tuple


def _is_stream_handle(data):
    obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
    return isinstance(obj_to_check, (BufferedIOBase, RawIOBase))


@functional_datapipe("load_from_tar_with_handler")
class TarArchiveLoaderWithHandler(TarArchiveLoader):
    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        handler: Callable = wds.reraise_exception,
        *args,
        **kwargs,
    ):
        super().__init__(datapipe, *args, **kwargs)
        self.handler = handler

        self.times = None
        self.profile = False

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            start = time.perf_counter()
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(
                    data_stream.file_obj, tarfile.TarFile
                ):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (
                        self.mode
                        if hasattr(data_stream, "seekable") and data_stream.seekable()
                        else self.mode.replace(":", "|")
                    )
                    # typing.cast is used here to silence mypy's type checker
                    tar = tarfile.open(
                        fileobj=cast(Optional[IO[bytes]], data_stream),
                        mode=reading_mode,
                        ignore_zeros=True,
                    )
                    if self.profile:
                        self.open_times.append(time.perf_counter() - start)
                try:
                    tarinfos = [tarinfo for tarinfo in tar]
                    tarinfos = sorted(tarinfos, key=lambda e: e.name)
                    for tarinfo in tarinfos:
                        start = time.perf_counter()
                        if not tarinfo.isfile():
                            continue
                        extracted_fobj = tar.extractfile(tarinfo)
                        if extracted_fobj is None:
                            warnings.warn(
                                f"failed to extract file {tarinfo.name} from source tarfile {pathname}"
                            )
                            raise tarfile.ExtractError
                        inner_pathname = os.path.normpath(
                            os.path.join(pathname, tarinfo.name)
                        )
                        sw = StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)  # type: ignore[misc]

                        if self.profile:
                            self.extract_times.append(time.perf_counter() - start)
                        yield inner_pathname, sw
                        # sw.autoclose()
                        del sw
                    # close tarfile after it's been exceeded
                finally:
                    tar.close()
                    del tar
                    del tarinfo

                    if _is_stream_handle(data_stream):
                        data_stream.autoclose()
                    del data_stream
                    gc.collect()
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!"
                )
                if self.handler(e):
                    if hasattr(e, "args") and len(e.args) > 0:
                        e.args = (e.args[0] + " @ " + str(pathname),) + e.args[1:]
                else:
                    raise e


@functional_datapipe("parse_jsonl_files_with_handler")
class JsonlParserIterDataPipe(IterDataPipe[Tuple[str, Dict]]):
    def __init__(
        self,
        datapipe: IterDataPipe[Tuple[str, IO]],
        handler: Callable = wds.reraise_exception,
        **kwargs,
    ) -> None:
        self.datapipe = datapipe
        self.kwargs = kwargs
        self.handler = handler

    def __iter__(self) -> Iterator[Tuple[str, Dict]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            for idx, line in enumerate(data_stream):
                if line.strip() != "":
                    try:
                        yield json.loads(line)
                    except Exception as e:
                        warnings.warn(
                            f"Error occured when parsing string to json due to: {e} abort!"
                        )
                        if self.handler(e):
                            if hasattr(e, "args") and len(e.args) > 0:
                                e.args = (e.args[0] + " @ " + str(pathname),) + e.args[
                                    1:
                                ]
                        else:
                            raise e
