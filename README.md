# DataML - Fused high-resolution evapotranspiration data
[Lu Li](https://www.researchgate.net/profile/Lu-Li-69?ev=hdr_xprf)

### Introduction
Evapotranspiration (ET) is the second largest hydrological flux over the land surface and connects water, energy, and carbon cycles. However, large uncertainties exist among current ET products due to their coarse spatial resolutions, short temporal coverages, and reliance on assumptions. This study introduces a multimodal machine learning framework to generate a high-resolution (0.1, daily), long-term (1950–2022) global ET dataset by fusing 13 state-of-the-art ET products encompassing remote sensing, machine learning, land surface models, and reanalysis data relying on extensive flux tower observations (462 sites). The framework reconstructs the individual ET products to consistent spatiotemporal resolutions and time ranges using Light Gradient Boosting Machine (LightGBM) models, and the Automated Machine Learning (AutoML) technique was used to fuse ET using 13 reconstructed ET products, ERA5-land atmospheric forcings and ancillary data as predictors. In-situ observations are utilized for model training and validation. Results demonstrate significant improvements over existing datasets, with our product achieving the highest accuracy (KGE =0.857, RMSE =0.726 mm/day) against in situ measurements across ecosystems and regions. The fused ET dataset realistically captures spatiotemporal variability and corrects the systematic underestimation bias prevalent in other datasets, particularly in wet regions. This novel high spatial-temporal ET dataset enables more robust assessments for water, energy, and carbon cycle applications on regional hydrology and ecology. The introduced data integration methodology also provides a valuable framework for fusing multiple geoscience datasets with disparate properties.

### Citation

In case you use DataML in your research or work, please cite:

```bibtex
@article{Lu Li,
    author = {Lu Li, Qingchen Xu, Zhongwang Wei et al.},
    title = {A multimodal machine learning fused global 0.1◦ daily evapotranspiration dataset from 1950-2022},
    journal = {Agricultural and Forest Meteorology},
    year = {2025},
    DOI = {https://doi.org/10.1016/j.agrformet.2025.110645}
}
```

### [License](https://github.com/leelew/DataML/LICENSE)
Copyright (c) 2023, Lu Li
