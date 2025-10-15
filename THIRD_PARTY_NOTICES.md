# Third-Party Notices

GeoWarp bundles or depends on the following third-party components. The original license texts are included in the distributed wheel or are accessible via the referenced upstream repositories.

| Component | Version (tested) | License | Source |
| --- | --- | --- | --- |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | 1.5+ | Apache License 2.0 | https://github.com/NVIDIA/warp/blob/main/LICENSE.txt |
| [NumPy](https://numpy.org/) | 1.24+ | BSD 3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| [Trimesh](https://trimsh.org/) | 4.5.1 | MIT License | https://github.com/mikedh/trimesh/blob/main/LICENSE.md |
| [Shapely](https://github.com/shapely/shapely) | 2.0+ | BSD 3-Clause | https://github.com/shapely/shapely/blob/main/LICENSE.txt |
| [PyEVTK](https://github.com/paulo-herrera/PyEVTK) (vendored) | 1.2.3 | MIT License | Included under `third_party/pyevtk` |
| [ImageIO](https://imageio.readthedocs.io/) | 2.35+ | BSD 2-Clause | https://github.com/imageio/imageio/blob/master/LICENSE |
| [Rich](https://github.com/Textualize/rich) | 13.9+ | MIT License | https://github.com/Textualize/rich/blob/master/LICENSE |
| [pynvml](https://github.com/gpuopenanalytics/pynvml) | 11.4+ | BSD 3-Clause | https://github.com/gpuopenanalytics/pynvml/blob/main/LICENSE |

Some example scripts refer to mesh assets (e.g., STL files) distributed under the same GPL-3.0 terms as GeoTaichi. Consult the upstream repositories for any additional usage restrictions when redistributing these assets.

If you introduce new dependencies, append them to this table with the verified version and license information. Ensure that each dependency's license is compatible with GPL-3.0 distribution.
