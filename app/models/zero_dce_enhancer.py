"""
Zero-DCE low-light image enhancement wrapper.

Mirrors the architecture from the training notebook:
  - DCE-Net: 7 Conv2D layers (32 filters, 3×3, stride 1, ReLU)
    with symmetrical skip connections and a final Tanh layer
    producing 24 parameter maps (8 iterations × 3 channels).
  - Enhancement: iterative curve application LE(x) = x + α·x·(1−x).

On startup the model attempts to load a Keras H5 model from
`models/zero_dce_model.h5`. If not found, it falls back to
a simple histogram-equalization placeholder so the pipeline
doesn't break while you upload the real weights.
"""

import logging
import os

import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("eyes.zero_dce")

# We only import TensorFlow if the model file exists to keep startup
# lightweight when the model hasn't been deployed yet.
_tf = None


def _lazy_import_tf():
    global _tf
    if _tf is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        _tf = tf
    return _tf


def _build_dce_net(image_size=256):
    """Rebuild the DCE-Net architecture so we can load .h5 weights."""
    tf = _lazy_import_tf()
    keras = tf.keras
    layers = keras.layers

    input_image = keras.Input(shape=[image_size, image_size, 3])
    conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(input_image)
    conv2 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv1)
    conv3 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv2)
    conv4 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(conv3)
    int_con1 = layers.Concatenate(axis=-1)([conv4, conv3])
    conv5 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con1)
    int_con2 = layers.Concatenate(axis=-1)([conv5, conv2])
    conv6 = layers.Conv2D(32, (3, 3), strides=(1, 1), activation="relu", padding="same")(int_con2)
    int_con3 = layers.Concatenate(axis=-1)([conv6, conv1])
    x_r = layers.Conv2D(24, (3, 3), strides=(1, 1), activation="tanh", padding="same")(int_con3)

    return keras.Model(inputs=input_image, outputs=x_r)


def _apply_enhancement(image_tensor, curve_params):
    """Apply iterative curve enhancement: LE(x) = x + α·x·(1−x), 8 iterations."""
    x = image_tensor
    for i in range(0, 3 * 8, 3):
        r = curve_params[:, :, :, i:i + 3]
        x = x + r * (x - x * x)
    return x


class ZeroDCEEnhancer:
    """Enhance low-light images using Zero-DCE (or fallback)."""

    def __init__(self):
        from app.config import settings

        self.dce_model = None
        model_path = settings.ZERO_DCE_MODEL_PATH

        if os.path.isfile(model_path):
            try:
                tf = _lazy_import_tf()
                # Try loading as a full Keras model first
                try:
                    self.dce_model = tf.keras.models.load_model(model_path, compile=False)
                    logger.info(f"Zero-DCE Keras model loaded from {model_path}")
                except Exception:
                    # If that fails, rebuild architecture and load weights only
                    self.dce_model = _build_dce_net()
                    self.dce_model.load_weights(model_path)
                    logger.info(f"Zero-DCE weights loaded into rebuilt DCE-Net from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Zero-DCE model: {e}. Using fallback.")
        else:
            logger.warning(
                f"Zero-DCE model not found at {model_path}. "
                "Using histogram-equalization fallback."
            )

    # ── public API ────────────────────────────────────────────

    def enhance(self, image: Image.Image) -> Image.Image:
        """Return an enhanced PIL Image."""
        if self.dce_model is not None:
            return self._enhance_with_model(image)
        return self._fallback_enhance(image)

    @staticmethod
    def is_low_light(image: Image.Image, threshold: float = 0.15) -> bool:
        """Heuristic: if average pixel brightness < threshold consider low-light."""
        gray = image.convert("L")
        mean_brightness = np.array(gray).mean() / 255.0
        return mean_brightness < threshold

    @staticmethod
    def get_brightness(image: Image.Image) -> float:
        """Return average brightness of the image as 0.0–1.0."""
        gray = image.convert("L")
        return float(np.array(gray).mean() / 255.0)

    # ── internals ─────────────────────────────────────────────

    def _enhance_with_model(self, image: Image.Image) -> Image.Image:
        tf = _lazy_import_tf()
        original_size = image.size

        img_array = np.array(image.resize((256, 256))).astype("float32") / 255.0
        img_array = img_array[:, :, :3]  # drop alpha if present
        input_tensor = tf.expand_dims(img_array, axis=0)

        # Get curve parameter maps from DCE-Net
        curve_params = self.dce_model(input_tensor)

        # Apply iterative curve enhancement
        enhanced = _apply_enhancement(input_tensor, curve_params)

        output_np = enhanced[0].numpy()
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)

        # Resize back to original dimensions
        result = Image.fromarray(output_np)
        result = result.resize(original_size, Image.LANCZOS)
        return result

    @staticmethod
    def _fallback_enhance(image: Image.Image) -> Image.Image:
        """Simple auto-contrast + brightness boost as a stand-in."""
        enhanced = ImageOps.autocontrast(image, cutoff=1)
        return enhanced
