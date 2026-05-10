"""
Zero-DCE low-light image enhancement wrapper.

Mirrors the architecture from the training notebook:
  - DCE-Net: 7 Conv2D layers (32 filters, 3×3, stride 1, ReLU)
    with symmetrical skip connections and a final Tanh layer
    producing 24 parameter maps (8 iterations × 3 channels).
  - Enhancement: iterative curve application LE(x) = x + α·x·(1−x).

Inference strategy (important for YOLO quality):
  - The DCE-Net predicts curve *parameters*, not pixel values. We
    therefore predict the curves on a fast 256×256 input but apply
    them at the original image resolution, preserving all the detail
    YOLO needs for detection. Resizing the enhanced pixels back up
    (as a prior version did) caused heavy blur and killed detections.

Safety behaviour:
  - Zero-DCE's tanh output can reduce already-bright regions. If a
    pass makes the image *darker* than it started, we discard it.
  - If the final result is still dim we stack the CLAHE + gamma
    fallback on top so YOLO always receives a usable frame.

On startup the model loads a Keras H5 file from
`models/zero_dce_model.h5`. If unavailable, a CLAHE + gamma +
auto-contrast fallback keeps the endpoint functional.
"""

import logging
import os

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

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


class ZeroDCEEnhancer:
    """Enhance low-light images using Zero-DCE (or fallback)."""

    # Minimum brightness we'd like YOLO to receive. Anything below this
    # after the model pass gets the classical boost stacked on top.
    _TARGET_BRIGHTNESS = 0.35

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
                "Using CLAHE + gamma fallback."
            )

    @property
    def has_model(self) -> bool:
        return self.dce_model is not None

    # ── public API ────────────────────────────────────────────

    def enhance(self, image: Image.Image, very_dark: bool = False) -> Image.Image:
        """Return an enhanced PIL Image guaranteed to be ≥ as bright as the input.

        Args:
            image: Input PIL image.
            very_dark: Allow an extra Zero-DCE pass + classical boost chain.
        """
        input_brightness = self.get_brightness(image)

        if self.dce_model is not None:
            result = self._enhance_with_model(image)
            out_brightness = self.get_brightness(result)
            logger.info(
                f"Brightness after 1st Zero-DCE pass: {out_brightness:.3f} "
                f"(input {input_brightness:.3f})"
            )

            # Safety: never return something darker than we got.
            if out_brightness < input_brightness:
                logger.info("Zero-DCE pass darkened the image → keeping original")
                result = image
                out_brightness = input_brightness

            # Optional 2nd pass for very dark inputs, but only if the 1st
            # pass actually helped. Otherwise we're just piling on noise.
            if very_dark and out_brightness < self._TARGET_BRIGHTNESS:
                second = self._enhance_with_model(result)
                b2 = self.get_brightness(second)
                logger.info(f"Brightness after 2nd Zero-DCE pass: {b2:.3f}")
                if b2 > out_brightness:
                    result = second
                    out_brightness = b2

            # Still too dim for detection? Stack classical boost on top.
            if out_brightness < self._TARGET_BRIGHTNESS:
                boosted = self._fallback_enhance(result, very_dark=very_dark)
                boosted_b = self.get_brightness(boosted)
                logger.info(
                    f"Brightness after CLAHE+gamma stack: {boosted_b:.3f}"
                )
                if boosted_b > out_brightness:
                    result = boosted

            return result

        # No model available — run the classical pipeline directly.
        return self._fallback_enhance(image, very_dark=very_dark)

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

    def _enhance_with_model(self, image: Image.Image) -> Image.Image:
        """Apply Zero-DCE curves at the *original* resolution.

        The DCE-Net requires a fixed 256×256 input, but the learned curve
        parameters are just 24 per-pixel coefficients. We therefore:
          1. Predict the 24-channel curve map at 256×256.
          2. Bilinearly upsample the curve map to the image's native
             resolution.
          3. Apply the 8-iteration LE(x)=x+α·x·(1−x) curve in-place on
             the full-resolution pixels.

        This preserves every pixel of detail YOLO needs while keeping
        inference cheap.
        """
        tf = _lazy_import_tf()

        # Full-resolution pixels (float32 in 0..1)
        img_full = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w = img_full.shape[:2]

        # Down-sample for the network
        img_small = np.asarray(image.convert("RGB").resize((256, 256), Image.BILINEAR),
                               dtype=np.float32) / 255.0
        input_tensor = tf.expand_dims(img_small, axis=0)

        # Predict curve parameter map at 256×256
        curve_params_small = self.dce_model(input_tensor).numpy()[0]  # (256, 256, 24)

        # Upsample curve params to native resolution (cv2 wants (w, h))
        curve_params_full = cv2.resize(
            curve_params_small, (w, h), interpolation=cv2.INTER_LINEAR
        )

        # Apply the 8-iteration Zero-DCE curve at full resolution
        x = img_full
        for i in range(0, 24, 3):
            r = curve_params_full[:, :, i:i + 3]
            x = x + r * (x - x * x)

        x = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(x)

    @staticmethod
    def _fallback_enhance(image: Image.Image, very_dark: bool = False) -> Image.Image:
        """Classical CLAHE → gamma → auto-contrast pipeline."""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clip_limit = 4.0 if very_dark else 3.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        img_cv = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        gamma = 0.4 if very_dark else 0.6
        inv_gamma = 1.0 / gamma
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        img_cv = cv2.LUT(img_cv, lut)

        enhanced = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        enhanced = ImageOps.autocontrast(enhanced, cutoff=1)

        if very_dark:
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.3)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)

        return enhanced
