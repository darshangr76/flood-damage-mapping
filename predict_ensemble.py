
"""
Flood Segmentation — Ensemble (ResNet34 + ResNet50)
"""
import os, glob
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
import cv2

class EnsembleFloodPredictor:
    def __init__(self, weights_r34, weights_r50, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_r34 = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=5, classes=1, activation=None
        ).to(self.device)
        self.model_r34.load_state_dict(torch.load(weights_r34, map_location=self.device))
        self.model_r34.eval()
        
        self.model_r50 = smp.Unet(
            encoder_name="resnet50", encoder_weights=None,
            in_channels=5, classes=1, activation=None
        ).to(self.device)
        self.model_r50.load_state_dict(torch.load(weights_r50, map_location=self.device))
        self.model_r50.eval()
    
    def _load_tif(self, folder, prefix):
        matches = glob.glob(os.path.join(folder, f"{prefix}*.tif"))
        if not matches:
            return np.zeros((224, 224), dtype=np.float32)
        with rasterio.open(matches[0]) as src:
            return src.read(1).astype(np.float32)
    
    def _to_db(self, x):
        return 10.0 * np.log10(np.clip(x, 1e-6, None))
    
    def _norm(self, x, vmin, vmax):
        return 2.0 * (np.clip(x, vmin, vmax) - vmin) / (vmax - vmin) - 1.0
    
    def _preprocess(self, folder):
        post_vv = self._norm(self._to_db(self._load_tif(folder, "MS1_IVV")), -25, 0)
        post_vh = self._norm(self._to_db(self._load_tif(folder, "MS1_IVH")), -30, -5)
        pre_vv  = self._norm(self._to_db(self._load_tif(folder, "SL1_IVV")), -25, 0)
        pre_vh  = self._norm(self._to_db(self._load_tif(folder, "SL1_IVH")), -30, -5)
        dem     = self._norm(self._load_tif(folder, "MK0_DEM"), 0, 5000)
        image = np.stack([pre_vv, pre_vh, post_vv, post_vh, dem], axis=0)
        return torch.from_numpy(image).float().unsqueeze(0).to(self.device)
    
    def predict_folder(self, folder, output_path, threshold=0.45, min_blob_area=50):
        ref_files = glob.glob(os.path.join(folder, "MS1_IVV*.tif"))
        if not ref_files:
            raise ValueError(f"No MS1_IVV file in {folder}")
        with rasterio.open(ref_files[0]) as ref:
            meta, ref_crs, ref_transform = ref.meta.copy(), ref.crs, ref.transform
        
        image_tensor = self._preprocess(folder)
        with torch.no_grad():
            p34 = torch.sigmoid(self.model_r34(image_tensor))
            p50 = torch.sigmoid(self.model_r50(image_tensor))
            prob = ((p34 + p50) / 2).squeeze().cpu().numpy()
        
        binary = (prob > threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                clean[labels == i] = 1
        
        meta.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw',
                     'crs': ref_crs, 'transform': ref_transform, 'nodata': 255})
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(clean, 1)
        
        return {
            "flood_pixels": int(clean.sum()),
            "total_pixels": int(clean.size),
            "flood_percentage": clean.sum() / clean.size * 100,
            "output_path": output_path,
            "method": "ensemble_resnet34_resnet50"
        }
