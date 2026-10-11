# AI Component Handoff - Person 2 to Person 1

## Model
- 3-model ensemble: ResNet18 + ResNet34 + ResNet50 (U-Net)
- Validation IoU: 0.7218
- Validation Recall: 0.8087
- Threshold: 0.45
- Input: 5 channels [pre_VV, pre_VH, post_VV, post_VH, DEM]

## Files
- predict_ensemble.py: main inference script
- kuro_v2_r18.pth, kuro_v2_r34.pth, kuro_v2_r50.pth: trained weights
- final_model_config.json: model configuration
- sample_masks/: sample GeoTIFF outputs for dashboard testing

## Usage
```python
from predict_ensemble import FloodEnsemble

predictor = FloodEnsemble()
result = predictor.predict_folder(
    'path/to/sentinel_sample_folder/',
    'output_flood_mask.tif'
)
print(result)
# {'flood_pixels': 13136, 'flood_percentage': 26.2, ...}
```

## Required Attribution (place in dashboard footer)
- Contains modified Copernicus Sentinel data 2026.
- (c) OpenStreetMap contributors.
- Trained on Kuro Siwo (Bountos et al., 2024).

## Known Limitations
1. Sentinel-1 revisit time: 6-12 days
2. SAR shadow in steep Himalayan valleys
3. Trained on Kuro Siwo 2015-2022 events, not 2026 Nepal flood
4. TTA with flips degraded performance (do not use)