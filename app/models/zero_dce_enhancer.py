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
    resolution, so no detail is lost. A prior version resized pixels
    back up, which blurred the output and hurt detections.

Aggressiveness:
  - Normal low-light: 8 curve iterations (the paper's 7-32-8 setting).
  - Very dark frames: 16 iterations (the paper's 7-32-16 setting) by
    cycling through the 8 learned curves twice. This is the strongest
    non-retrained variant reported in the paper.

Safety guarantees (important – we cannot return a darker image):
  - Every stage is monitored. If the candidate output is darker than
    the stage input, we discard it.
  - A final classical boost (CLAHE + correct gamma) is stacked on top
    if brightness is still below the detection-friendly target so
    YOLO always receives a usable frame.
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
    """Enhance low-light images using Zero-DCE (with classical safety net)."""

    # Target brightness YOLO expects. Anything below this after the model
    # pass gets the classical boost stacked on top.
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
                "Using CLAHE + gamma fallback."
            )

    @property
    def has_model(self) -> bool:
        return self.dce_model is not None

    # ── public API ────────────────────────────────────────────

    def enhance(self, image: Image.Image, very_dark: bool = False) -> Image.Image:
        """Return an enhanced PIL Image, guaranteed ≥ as bright as the input."""
        input_brightness = self.get_brightness(image)

        # Step 1 – Zero-DCE model pass (16 iterations for very dark, 8 otherwise)
        if self.dce_model is not None:
            iterations = 16 if very_dark else 8
            result = self._enhance_with_model(image, iterations=iterations)
            out_brightness = self.get_brightness(result)
            logger.info(
                f"Zero-DCE ({iterations} iter): {input_brightness:.3f} → {out_brightness:.3f}"
            )
            # Never let the model pass make the image darker.
            if out_brightness < input_brightness:
                logger.info("Zero-DCE pass darkened image → keeping original")
                result = image
                out_brightness = input_brightness
        else:
            result = image
            out_brightness = input_brightness

        # Step 2 – Classical boost stacked on top if still too dim for detection.
        if out_brightness < self._TARGET_BRIGHTNESS:
            boosted = self._fallback_enhance(result, very_dark=very_dark)
            boosted_brightness = self.get_brightness(boosted)
            logger.info(
                f"Classical boost: {out_brightness:.3f} → {boosted_brightness:.3f}"
            )
            if boosted_brightness > out_brightness:
                result = boosted
                out_brightness = boosted_brightness

        # Step 3 – Last-resort linear brightness scale. Guarantees the caller
        # sees *something* brighter than the input for truly under-exposed
        # frames. This only kicks in if every preceding stage left us dim.
        if out_brightness < self._TARGET_BRIGHTNESS and out_brightness > 0:
            scale = min(self._TARGET_BRIGHTNESS / out_brightness, 4.0)
            arr = np.asarray(result.convert("RGB"), dtype=np.float32) * scale
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            result = Image.fromarray(arr)
            logger.info(
                f"Linear rescale ×{scale:.2f} → {self.get_brightness(result):.3f}"
            )

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

    @staticmethod
    def _fallback_enhance(image: Image.Image, very_dark: bool = False) -> Image.Image:
        """CLAHE → gamma (correct direction) → auto-contrast → brightness/contrast."""
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # ── Stage 1: CLAHE on L channel (LAB) for adaptive local contrast. ──
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clip_limit = 6.0 if very_dark else 3.5
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        img_cv = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # ── Stage 2: Gamma correction. out = in**(1/gamma). ─────────────────
        # IMPORTANT: gamma > 1 brightens (pulls midtones up); gamma < 1 darkens.
        # A prior version used gamma < 1 here, which silently *darkened* the
        # image and defeated the entire enhancer.
        gamma = 2.8 if very_dark else 1.8
        inv_gamma = 1.0 / gamma
        lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")
        img_cv = cv2.LUT(img_cv, lut)

        enhanced = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        # ── Stage 3: Auto-contrast so we use the full tonal range. ──────────
        enhanced = ImageOps.autocontrast(enhanced, cutoff=1)

        # ── Stage 4: Small brightness + contrast polish for very dark input. ─
        if very_dark:
            enhanced = ImageEnhance.Brightness(enhanced).enhance(1.4)
            enhanced = ImageEnhance.Contrast(enhanced).enhance(1.25)

        return enhanced
