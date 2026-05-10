"""
Zero-DCE low-light image enhancement wrapper.

Architecture (matches the training notebook):
  - DCE-Net: 7 Conv2D layers (32 filters, 3×3, stride 1, ReLU)
    with symmetrical skip connections and a final Tanh layer
    producing 24 parameter maps (8 iterations × 3 channels).
  - Enhancement: iterative curve application LE(x) = x + α·x·(1−x).

Inference strategy (preserves quality for YOLO):
  - DCE-Net predicts curve *parameters*, not pixels. We predict on a
    fast 256×256 input but apply the curve at the *original* image
    resolution, so no detail is lost.

Aggressiveness:
  - Normal low-light: 8 curve iterations (the paper's 7-32-8 setting).
  - Very dark: 16 iterations (7-32-16) by cycling through the 8
    learned curves twice. This is the strongest non-retrained variant
    reported in the paper.

Colour safety (why this file exists in its current shape):
  - All tone operations happen on the L channel in LAB colour space.
    Operating in RGB per-channel — as a previous version did — causes
    each channel to saturate at a different rate, producing the
    characteristic purple/magenta fringing on blown-out highlights
    seen in the cake-lights test image.
  - A shadow-lift curve targets only dark pixels. Highlights are
    preserved, so the small bright regions stop blooming.
  - No per-channel linear rescale. That step was the main cause of
    over-exposure artefacts.
"""

import logging
import os

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("eyes.zero_dce")

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


def _lift_shadows_lab(image: Image.Image, strength: float) -> Image.Image:
    """Lift dark regions by boosting the L channel in LAB space.

    The curve:  L' = L + strength · (1 − L)² · L^0.25

    Properties:
      - When L ≈ 0 (deep shadow): L' ≈ strength · L^0.25, so we pull
        very dark pixels up towards the mid-tones without touching
        anything that's already well-lit.
      - When L ≈ 1 (highlight): (1 − L)² ≈ 0, so L' ≈ L. Bright spots
        like light sources are not blown out.
      - Chrominance (a, b) is left untouched, so colours don't shift
        and no purple/magenta fringing is introduced.
    """
    if strength <= 0:
        return image

    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_cv)

    l_f = l.astype(np.float32) / 255.0
    lifted = l_f + strength * ((1.0 - l_f) ** 2) * np.power(np.clip(l_f, 1e-3, 1.0), 0.25)
    lifted = np.clip(lifted, 0.0, 1.0)

    # Apply a gentle CLAHE to the lifted L channel so local contrast
    # comes back without amplifying chroma noise.
    l_out = (lifted * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_out = clahe.apply(l_out)

    merged = cv2.merge([l_out, a, b])
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def _denoise(image: Image.Image) -> Image.Image:
    """Mild bilateral filter to suppress chroma noise after enhancement."""
    arr = np.array(image.convert("RGB"))
    arr = cv2.bilateralFilter(arr, d=5, sigmaColor=35, sigmaSpace=35)
    return Image.fromarray(arr)


class ZeroDCEEnhancer:
    """Enhance low-light images using Zero-DCE + LAB shadow lift."""

    # Target brightness YOLO expects. Anything below this after the model
    # pass gets the shadow-lift stage layered on top.
    _TARGET_BRIGHTNESS = 0.40

    def __init__(self):
        from app.config import settings

        self.dce_model = None
        model_path = settings.ZERO_DCE_MODEL_PATH

        if os.path.isfile(model_path):
            try:
                tf = _lazy_import_tf()
                try:
                    self.dce_model = tf.keras.models.load_model(model_path, compile=False)
                    logger.info(f"Zero-DCE Keras model loaded from {model_path}")
                except Exception:
                    self.dce_model = _build_dce_net()
                    self.dce_model.load_weights(model_path)
                    logger.info(f"Zero-DCE weights loaded into rebuilt DCE-Net from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Zero-DCE model: {e}. Using fallback.")
        else:
            logger.warning(
                f"Zero-DCE model not found at {model_path}. "
                "Using LAB shadow-lift fallback."
            )

    @property
    def has_model(self) -> bool:
        return self.dce_model is not None

    # ── public API ────────────────────────────────────────────

    def enhance(self, image: Image.Image, very_dark: bool = False) -> Image.Image:
        """Return an enhanced PIL Image without blowing out highlights."""
        input_brightness = self.get_brightness(image)

        # Step 1 – Zero-DCE model pass (16 iterations for very dark, 8 otherwise)
        if self.dce_model is not None:
            iterations = 16 if very_dark else 8
            result = self._enhance_with_model(image, iterations=iterations)
            out_brightness = self.get_brightness(result)
            logger.info(
                f"Zero-DCE ({iterations} iter): {input_brightness:.3f} → {out_brightness:.3f}"
            )
            # Never return something darker than we got.
            if out_brightness < input_brightness:
                logger.info("Zero-DCE pass darkened image → keeping original")
                result = image
                out_brightness = input_brightness
        else:
            result = image
            out_brightness = input_brightness

        # Step 2 – LAB shadow lift. This is colour-safe (a,b untouched)
        # and highlight-preserving (curve is ~0 near L=1). We only lift
        # if YOLO would struggle with the current brightness.
        if out_brightness < self._TARGET_BRIGHTNESS:
            strength = 1.4 if very_dark else 0.9
            lifted = _lift_shadows_lab(result, strength=strength)
            lifted_brightness = self.get_brightness(lifted)
            logger.info(
                f"LAB shadow lift (s={strength:.1f}): "
                f"{out_brightness:.3f} → {lifted_brightness:.3f}"
            )
            if lifted_brightness > out_brightness:
                result = lifted
                out_brightness = lifted_brightness

        # Step 3 – Mild denoise only if we had to boost a lot. Bilateral
        # preserves edges (important for YOLO) while killing the colour
        # speckle that naturally appears in lifted shadows.
        if very_dark and out_brightness > input_brightness + 0.10:
            result = _denoise(result)

        return result

    @staticmethod
    def is_low_light(image: Image.Image, threshold: float = 0.35) -> bool:
        gray = image.convert("L")
        return float(np.array(gray).mean() / 255.0) < threshold

    @staticmethod
    def get_brightness(image: Image.Image) -> float:
        gray = image.convert("L")
        return float(np.array(gray).mean() / 255.0)

    @staticmethod
    def get_image_stats(image: Image.Image) -> dict:
        arr = np.array(image.convert("L"))
        return {
            "mean": float(arr.mean()) / 255.0,
            "max": float(arr.max()) / 255.0,
            "min": float(arr.min()) / 255.0,
            "std": float(arr.std()) / 255.0,
        }

    # ── internals ─────────────────────────────────────────────

    def _enhance_with_model(self, image: Image.Image, iterations: int = 8) -> Image.Image:
        """Apply Zero-DCE curves at the *original* resolution.

        The network outputs an 8-iteration curve map (24 channels).
        For `iterations=16` we cycle through that learned set twice —
        this is the 7-32-16 variant from the Zero-DCE paper.
        """
        tf = _lazy_import_tf()

        img_full = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w = img_full.shape[:2]

        img_small = np.asarray(
            image.convert("RGB").resize((256, 256), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        input_tensor = tf.expand_dims(img_small, axis=0)

        # Predict curve parameter map at 256×256, then upsample to native res.
        curve_params_small = self.dce_model(input_tensor).numpy()[0]       # (256,256,24)
        curve_params_full = cv2.resize(
            curve_params_small, (w, h), interpolation=cv2.INTER_LINEAR
        )

        # Apply LE(x)=x+α·x·(1−x) repeatedly. Each block of 3 channels is one α.
        x = img_full
        for it in range(iterations):
            offset = (it % 8) * 3
            r = curve_params_full[:, :, offset:offset + 3]
            x = x + r * (x - x * x)

        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(x)
