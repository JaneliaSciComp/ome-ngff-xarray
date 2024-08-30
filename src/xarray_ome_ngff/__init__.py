from __future__ import annotations
from typing import TYPE_CHECKING, Any
from importlib.metadata import version as _version
from zarr.storage import BaseStore
from xarray_ome_ngff.array_wrap import (
    ArrayWrapperSpec,
    DaskArrayWrapper,
    ZarrArrayWrapper,
)
from xarray_ome_ngff.core import NGFF_VERSIONS

if TYPE_CHECKING:
    from typing import Literal
    from pydantic_ome_ngff.v04.multiscale import MultiscaleGroup as MultiscaleGroupV04
from numcodecs.abc import Codec
from xarray import DataArray
import zarr
from xarray_ome_ngff.v04 import multiscale as multiscale_v04

__version__ = _version(__name__)


# todo: remove the need to specify the version for reading
def read_multiscale_group(
    group: zarr.Group,
    array_wrapper: ZarrArrayWrapper
    | DaskArrayWrapper
    | ArrayWrapperSpec = ZarrArrayWrapper(),
    ngff_version: Literal["0.4"] = "0.4",
    **kwargs,
) -> dict[str, DataArray]:
    """
    Create a dictionary of `xarray.DataArray` from a Zarr group that implements some version of the
    OME-NGFF multiscale image specification.

    The keys of the dictionary are the paths to the Zarr arrays. The values of the dictionary are
    `xarray.DataArray` objects, one per Zarr array described in the OME-NGFF multiscale metadata.

    Parameters
    ----------
    group: zarr.Group
        A handle for the Zarr group that contains the OME-NGFF metadata.
    array_wrapper: ZarrArrayWrapper | DaskArrayWrapper | ArrayWrapperSpec, default = ZarrArrayWrapper
        Either an object that implements `BaseArrayWrapper`, or a dict model of such a subclass,
        which will be resolved to an object implementing `BaseArrayWrapper`. This object has a
        `wrap` method that takes an instance of `zarr.Array` and returns another array-like object.
        This enables wrapping Zarr arrays in a lazy array representation like Dask arrays
        (e.g., via `DaskArrayWrapper), which is necessary when working with large Zarr arrays.
    **kwargs: Any
        Additional keyword arguments that will be passed to the `read_group` function defined for a
        specific OME-NGFF version
    """
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.read_multiscale_group(
            group, array_wrapper=array_wrapper, **kwargs
        )


# todo: remove the need to specify the version for reading
def read_multiscale_array(
    array: zarr.Array,
    array_wrapper: ZarrArrayWrapper
    | DaskArrayWrapper
    | ArrayWrapperSpec = ZarrArrayWrapper(),
    ngff_version: Literal["0.4"] = "0.4",
    **kwargs,
) -> DataArray:
    """
    Read a single Zarr array as an `xarray.DataArray`, using OME-NGFF
    metadata.

    Parameters
    ----------
    array: zarr.Array
        A Zarr array that is part of a version 0.4 OME-NGFF multiscale image.
    array_wrapper: BaseArrayWrapper | ArrayWrapperSpec, default is ZarrArrayWrapper
        The array wrapper class to use when converting the Zarr array to an `xarray.DataArray`.

    Returns
    -------
    xarray.DataArray
    """
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.read_multiscale_array(
            array, array_wrapper=array_wrapper, **kwargs
        )


def model_multiscale_group(
    *,
    arrays: dict[str, DataArray],
    transform_precision: int | None = None,
    ngff_version: Literal["0.4"] = "0.4",
    chunks: tuple[int, ...] | tuple[tuple[int, ...]] | Literal["auto"] = "auto",
    compressor: Codec | None = multiscale_v04.DEFAULT_COMPRESSOR,
    fill_value: Any = 0,
) -> MultiscaleGroupV04:
    """
    Create a model of an OME-NGFF multiscale group from a dict of `xarray.DataArray`.
    The dimensions / coordinates of the arrays will be used to infer OME-NGFF axis metadata, as well
    as the OME-NGFF coordinate transformation metadata (i.e., scaling and translation).

    Parameters
    ----------
    arrays: dict[str, DataArray]
        A mapping from strings to `xarray.DataArray`.
    transform_precision: int | None, default is None
        Whether, and how much, to round the transformations estimated from the coordinates.
        The default (`None`) results in no rounding; specifying an `int` x will round transforms to
        x decimal places using `numpy.round(transform, x)`.
    ngff_version: Literal["0.4"]
        The OME-NGFF version to use.
    chunks: tuple[int] | tuple[tuple[int, ...]] | Literal["auto"], default = "auto"
        The chunks for the arrays in the multiscale group.
        If the string "auto" is provided, each array will have chunks set to the zarr-python default
        value, which depends on the shape and dtype of the array.
        If a single sequence of ints is provided, then this defines the chunks for all arrays.
        If a sequence of sequences of ints is provided, then this defines the chunks for each array.
    compressor: Codec | None, default = numcodecs.ZStd.
        The compressor to use for the arrays. Default is `numcodecs.ZStd`.
    fill_value: Any
        The fill value for the Zarr arrays.
    """
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.model_multiscale_group(
            arrays=arrays,
            transform_precision=transform_precision,
            compressor=compressor,
            chunks=chunks,
            fill_value=fill_value,
        )


def create_multiscale_group(
    *,
    store: BaseStore,
    path: str,
    arrays: dict[str, DataArray],
    transform_precision: int | None = None,
    ngff_version: Literal["0.4"] = "0.4",
    chunks: tuple[int, ...] | tuple[tuple[int, ...]] | Literal["auto"] = "auto",
    compressor: Codec | None = multiscale_v04.DEFAULT_COMPRESSOR,
    fill_value: Any = 0,
    overwrite: bool = False,
) -> MultiscaleGroupV04:
    """
    store: zarr.storage.BaseStore
        The storage backend for the Zarr hierarchy.
    path: str
        The path in the storage backend for the multiscale group.
    arrays: dict[str, DataArray]
        A mapping from strings to `xarray.DataArray`.
    transform_precision: int | None, default is None
        Whether, and how much, to round the transformations estimated from the coordinates.
        The default (`None`) results in no rounding; specifying an `int` x will round transforms to
        x decimal places using `numpy.round(transform, x)`.
    ngff_version: Literal["0.4"]
        The OME-NGFF version to use.
    chunks: tuple[int] | tuple[tuple[int, ...]] | Literal["auto"], default = "auto"
        The chunks for the arrays in the multiscale group.
        If the string "auto" is provided, each array will have chunks set to the zarr-python default
        value, which depends on the shape and dtype of the array.
        If a single sequence of ints is provided, then this defines the chunks for all arrays.
        If a sequence of sequences of ints is provided, then this defines the chunks for each array.
    compressor: Codec | None, default = numcodecs.ZStd.
        The compressor to use for the arrays. Default is `numcodecs.ZStd`.
    fill_value: Any
        The fill value for the Zarr arrays.
    """
    if ngff_version not in NGFF_VERSIONS:
        raise ValueError(f"Unsupported NGFF version: {ngff_version}")
    if ngff_version == "0.4":
        return multiscale_v04.create_multiscale_group(
            store=store,
            path=path,
            arrays=arrays,
            transform_precision=transform_precision,
            compressor=compressor,
            chunks=chunks,
            fill_value=fill_value,
            overwrite=overwrite,
        )


__all__ = [
    "read_multiscale_array",
    "read_multiscale_array",
    "create_multiscale_group",
    "model_multiscale_group",
    "DaskArrayWrapper",
    "ZarrArrayWrapper",
]
