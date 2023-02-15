from xarray import DataArray
import numpy as np

from xarray_ome_ngff.v05.multiscales import (
    create_axes_transforms,
    create_multiscale_metadata,
)
from pydantic_ome_ngff.v05.multiscales import Multiscale, MultiscaleDataset


def create_array(shape, axes, units, types, scale, translate, **kwargs):
    """
    Create a dataarray with a shape and coordinates
    defined by the parameters axes, units, types, scale, translate.
    """
    coords = []
    for ax, u, sh, sc, tr, ty in zip(axes, units, shape, scale, translate, types):
        coords.append(
            DataArray(
                (np.arange(sh) * sc) + tr, dims=(ax), attrs={"units": u, "type": ty}
            )
        )

    return DataArray(np.zeros(shape), coords=coords, **kwargs)


def test_ome_ngff_from_arrays():
    axes = ("z", "y", "x")
    units = ("meter", "meter", "meter")
    types = ("space", "space", "space")
    translate = (0, -8, 10)
    scale = (1.0, 1.0, 10.0)
    shape = (16,) * 3
    data = create_array(shape, axes, units, types, scale, translate)
    coarsen_kwargs = {**{dim: 2 for dim in axes}, "boundary": "trim"}
    multi = [data, data.coarsen(**coarsen_kwargs).mean()]
    multi.append(multi[-1].coarsen(**coarsen_kwargs).mean())
    for idx, m in enumerate(multi):
        m.name = f"s{idx}"
    axes, transforms = tuple(zip(*(create_axes_transforms(m) for m in multi)))
    multiscale_meta = create_multiscale_metadata(multi).dict()
    expected_meta = Multiscale(
        name=None,
        version="0.5-dev",
        type=None,
        metadata=None,
        datasets=[
            MultiscaleDataset(path=m.name, coordinateTransformations=transforms[idx])
            for idx, m in enumerate(multi)
        ],
        axes=axes[0],
    ).dict()

    assert multiscale_meta == expected_meta
