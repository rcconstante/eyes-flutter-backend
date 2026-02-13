# Model Weights Directory

Place your trained model files here before deployment:

| File | Description |
|------|-------------|
| `yolov8n.pt` | YOLOv8n custom-trained weights (from `yolo_model_training.py`) |
| `zero_dce_model.h5` | Keras H5 model weights (from `zero_reference_dce.py`) |

**MiDaS** weights are auto-downloaded from PyTorch Hub on first run.

## How to export from training notebooks

### YOLOv8
After training, the best weights are saved at `runs/detect/train/weights/best.pt`.
Copy that file here as `yolov8n.pt`.

### Zero-DCE
From the training notebook, the model is exported via:
```python
zero_dce_model.dce_model.save("zero_dce_model.h5")
```
Copy the `zero_dce_model.h5` file here.
