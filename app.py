"""Streamlit deployment dashboard for the Taswira plane finder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from deployment import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    best_frames,
    load_config,
    load_interpreter,
    predict_frame,
    predict_sweep,
    sweep_summary,
)


st.set_page_config(page_title="Taswira Plane Finder", page_icon="🩺", layout="wide")


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_resource(show_spinner="Loading TensorFlow Lite model…")
def get_interpreter(model_path: str):
    return load_interpreter(model_path)


def artifact_status() -> tuple[bool, str]:
    missing = [str(path) for path in (DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH) if not path.exists()]
    return (not missing, "" if not missing else "Missing: " + ", ".join(missing))


def render_prediction(result: dict, config: dict) -> None:
    predicted = result["predicted_class"]
    confidence = result["confidence"]
    if predicted == "no_plane":
        st.warning(f"No standard plane detected ({confidence:.1%} confidence).")
    else:
        st.success(f"Detected view: {predicted.replace('_', ' ').title()} ({confidence:.1%} confidence)")
    chart = pd.DataFrame({"class": list(result["probabilities"]), "probability": list(result["probabilities"].values())})
    st.bar_chart(chart.set_index("class"), y="probability", height=260)


def main() -> None:
    st.title("Taswira Plane Finder")
    st.caption("Ultrasound plane-finder deployment dashboard")
    st.info("Decision-support demonstration only. This model does not diagnose, replace a sonographer, or determine referral by itself.")

    ready, status = artifact_status()
    with st.sidebar:
        st.header("Deployment")
        if ready:
            st.success("Model artifacts found")
        else:
            st.error(status)
        st.caption("Canonical artifacts")
        st.code("outputs/taswira_plane_finder.tflite\noutputs/pipeline_config.json")
        st.divider()
        st.write("Supported views")
        st.write("biparietal · abdominal · heart · spine · femur · no plane")

    if not ready:
        st.stop()
    try:
        config = get_config()
        interpreter = get_interpreter(str(DEFAULT_MODEL_PATH))
    except Exception as exc:
        st.error(f"The deployment model could not be loaded: {exc}")
        st.stop()

    single_tab, sweep_tab, info_tab, breech_tab, anomaly_tab = st.tabs(
        ["Single image", "Sweep analysis", "Model information", "Breech screening", "Brain anomaly screening"]
    )

    with single_tab:
        st.subheader("Analyze one ultrasound frame")
        upload = st.file_uploader("Upload a frame", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"], key="single")
        if upload:
            try:
                image = Image.open(upload).convert("RGB")
                left, right = st.columns([1, 1])
                with left:
                    st.image(image, caption=upload.name, use_container_width=True)
                with right:
                    with st.spinner("Running inference…"):
                        result = predict_frame(interpreter, image, config)
                    render_prediction(result, config)
            except Exception as exc:
                st.error(f"Could not analyze this image: {exc}")

    with sweep_tab:
        st.subheader("Analyze multiple frames")
        uploads = st.file_uploader(
            "Upload frames from one sweep",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="sweep",
        )
        if uploads:
            try:
                images = [Image.open(item).convert("RGB") for item in uploads]
                with st.spinner(f"Analyzing {len(images)} frames…"):
                    results = predict_sweep(interpreter, images, config)
                summary = sweep_summary(results, config)
                if summary["usable"]:
                    st.success(f"Sweep usable: best plane score {summary['best_score']:.1%} ≥ {summary['threshold']:.1%}")
                else:
                    st.warning(f"Sweep may need repeating: best plane score {summary['best_score']:.1%} < {summary['threshold']:.1%}")
                st.metric("Frames analyzed", summary["frames"])
                best = best_frames(results, config)
                st.subheader("Best frame per configured view")
                cols = st.columns(len(config["classes"]))
                for column, class_name in zip(cols, config["classes"]):
                    row = best[class_name]
                    with column:
                        st.metric(class_name.replace("_", " ").title(), f"{row['probabilities'][class_name]:.1%}", f"frame {row['frame']}")
                table = pd.DataFrame([
                    {"frame": row["frame"], "prediction": row["predicted_class"], "confidence": row["confidence"]}
                    for row in results
                ])
                st.dataframe(table, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Could not analyze this sweep: {exc}")

    with info_tab:
        st.subheader("Deployment model")
        st.json({
            "model": "TensorFlow Lite",
            "backbone": config.get("backbone"),
            "head": config.get("head"),
            "image_size": config["img_size"],
            "classes": config["classes"],
            "frame_threshold": config["frame_threshold"],
            "sweep_threshold": config["sweep_threshold"],
        })
        st.write("Preprocessing uses the configured crop box, grayscale conversion, percentile contrast stretching, square padding, and RGB channel replication.")

    with breech_tab:
        st.subheader("Breech screening")
        st.warning("Not available in this deployment build.")
        st.write("The breech classifier exists only as notebook analysis and has not been exported as a standalone model with deployment metadata.")

    with anomaly_tab:
        st.subheader("Brain anomaly screening")
        st.warning("Not available in this deployment build.")
        st.write("The anomaly workflow is exploratory notebook code and has not been packaged as a standalone deployment model. No anomaly result is generated here.")


if __name__ == "__main__":
    main()
