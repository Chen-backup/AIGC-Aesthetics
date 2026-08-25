from __future__ import annotations

import sys
from pathlib import Path


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))

    from _fusion20d_common import run_fusion_20d_model

    run_fusion_20d_model(
        ai_model_type="DINOv2",
        ai_features_path=parent_dir / "BYS_DINOv2_Features_Model" / "PCA_10_dinov2.csv",
        output_dir=current_dir / "Result",
    )
