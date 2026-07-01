import torch

# PyTorch 2.6+ defaults torch.load to weights_only=True, which the old
# ultralytics==8.0.239 (pulled in by ultralyticsplus) doesn't know how to
# handle. We trust this checkpoint's source (Hugging Face), so force the
# old, permissive loading behavior just for this one-time download.
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralyticsplus import YOLO

model = YOLO('keremberke/yolov8n-license-plate')
model.save('weights/license_plate_yolov8n.pt')
