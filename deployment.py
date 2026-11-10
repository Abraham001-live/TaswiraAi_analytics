"""Runtime helpers for the packaged Taswira plane-finder model.

This module deliberately contains no notebook state and no model training code.
It is safe to import from tests or from the Streamlit application.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "outputs" / "taswira_plane_finder.tflite"
DEFAULT_CONFIG_PATH = PROJECT_DIR / "outputs" / "pipeline_config.json"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and lightly validate the deployment configuration."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"classes", "img_size", "crop_box", "contrast_percentiles", "frame_threshold", "sweep_threshold"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Deployment configuration is missing: {', '.join(sorted(missing))}")
    if len(config["crop_box"]) != 4 or len(config["contrast_percentiles"]) != 2:
        raise ValueError("crop_box must have four values and contrast_percentiles must have two values")
    return config


def preprocess_image(image: Image.Image, config: dict[str, Any]) -> np.ndarray:
    """Match the notebook preprocessing and return a float32 RGB model tensor.

    The model was trained from grayscale images repeated across three channels.
    """
    gray = image.convert("L")
    width, height = gray.size
    left, top, right, bottom = [int(value) for value in config["crop_box"]]
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        raise ValueError("The configured crop box does not overlap the uploaded image")

    array = np.asarray(gray.crop((left, top, right, bottom)), dtype=np.float32)
    low_pct, high_pct = config["contrast_percentiles"]
    low, high = np.percentile(array, [low_pct, high_pct])
    array = np.clip((array - low) / max(float(high - low), 1e-6) * 255.0, 0, 255).astype(np.uint8)

    side = max(array.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    y_offset = (side - array.shape[0]) // 2
    x_offset = (side - array.shape[1]) // 2
    square[y_offset:y_offset + array.shape[0], x_offset:x_offset + array.shape[1]] = array
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = Image.fromarray(square).resize((int(config["img_size"]), int(config["img_size"])), resampling)
    rgb = np.repeat(np.asarray(resized, dtype=np.float32)[..., None], 3, axis=-1)
    return rgb[None, ...]


def load_interpreter(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Create an allocated TensorFlow Lite interpreter on demand."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def predict_frame(interpreter: Any, image: Image.Image, config: dict[str, Any]) -> dict[str, Any]:
    """Run one image and return a JSON-friendly prediction record."""
    tensor = preprocess_image(image, config)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    expected_shape = tuple(int(value) for value in input_details["shape"])
    if expected_shape != tuple(tensor.shape):
        raise ValueError(f"Model expects input shape {expected_shape}, but preprocessing produced {tuple(tensor.shape)}")
    model_input = tensor
    input_scale, input_zero = input_details.get("quantization", (0.0, 0))
    if np.issubdtype(input_details["dtype"], np.integer) and input_scale:
        model_input = np.round(tensor / input_scale + input_zero)
        info = np.iinfo(input_details["dtype"])
        model_input = np.clip(model_input, info.min, info.max)
    interpreter.set_tensor(input_details["index"], model_input.astype(input_details["dtype"], copy=False))
    interpreter.invoke()
    probabilities = np.asarray(interpreter.get_tensor(output_details["index"])[0], dtype=np.float64)
    output_scale, output_zero = output_details.get("quantization", (0.0, 0))
    if np.issubdtype(output_details["dtype"], np.integer) and output_scale:
        probabilities = (probabilities - output_zero) * output_scale
    if probabilities.ndim != 1 or len(probabilities) != len(config["classes"]):
        raise ValueError("Model output does not match the configured class list")
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite probabilities")
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("Model returned an invalid probability vector")
    probabilities = probabilities / total
    index = int(probabilities.argmax())
    classes = config["classes"]
    return {
        "predicted_class": classes[index],
        "confidence": float(probabilities[index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(classes)},
    }


def predict_sweep(interpreter: Any, images: Iterable[Image.Image], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Predict a sequence and annotate each record with its source frame index."""
    results = []
    for index, image in enumerate(images, start=1):
        result = predict_frame(interpreter, image, config)
        result["frame"] = index
        results.append(result)
    return results


def best_frames(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the highest-probability frame for each configured class."""
    selected: dict[str, dict[str, Any]] = {}
    for class_name in config["classes"]:
        candidates = [row for row in results if class_name in row["probabilities"]]
        if candidates:
            selected[class_name] = max(candidates, key=lambda row: row["probabilities"][class_name])
    return selected


def sweep_summary(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Summarize whether the sequence passes the configured sweep threshold."""
    if not results:
        return {"usable": False, "best_score": 0.0, "threshold": float(config["sweep_threshold"]), "frames": 0}
    no_plane = config["classes"].index("no_plane")
    best_score = max(1.0 - row["probabilities"]["no_plane"] for row in results)
    threshold = float(config["sweep_threshold"])
    return {"usable": best_score >= threshold, "best_score": best_score, "threshold": threshold, "frames": len(results)}
