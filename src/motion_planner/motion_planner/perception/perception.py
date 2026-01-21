from numpy import np
from ultralytics import YOLOE

class PerceptionModel:
    def __init__(self):
        
        self.yolo = YOLOE("yoloe-11s-seg.pt")
        self.yolo.to("cuda:0")
        
    def inference(self, img, depth_img, intrinsics, ref_img=None, bbox=None, tp=None):
        
        return None

    def calculate_object_center_3d(self, segmentation_mask, depth_image, intrinsics, unit_divisor=1000.0):
        """
        Computes 3D center using percentile filtering.
        """
        if not np.any(segmentation_mask):
            return None

        rows, cols = np.where(segmentation_mask)
        u_cx, v_cy = np.mean(cols), np.mean(rows)

        raw_depths = depth_image[segmentation_mask]
        valid_mask = (np.isfinite(raw_depths)) & (raw_depths > 0)
        valid_raw_depths = raw_depths[valid_mask]

        if valid_raw_depths.size == 0:
            return None

        depths_m = valid_raw_depths / unit_divisor

        # Z Calculation (5th-70th Percentile)
        d_min, d_max = np.percentile(depths_m, [5, 70])
        clean_depths = depths_m[(depths_m >= d_min) & (depths_m <= d_max)]
        
        z_center = np.median(clean_depths) if clean_depths.size > 0 else np.median(depths_m)

        x_center = (u_cx - intrinsics['cx']) * z_center / intrinsics['fx']
        y_center = (v_cy - intrinsics['cy']) * z_center / intrinsics['fy']

        return (x_center, y_center, z_center)
