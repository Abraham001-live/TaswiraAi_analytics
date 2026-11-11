"""Plain-language Streamlit interface for the Taswira plane finder."""

from __future__ import annotations

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

st.set_page_config(page_title="Taswira Ultrasound Assistant", page_icon="🩺", layout="wide")
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]


def friendly_name(value: str) -> str:
    names = {"biparietal": "Head (biparietal)", "abdominal": "Abdomen", "heart": "Heart", "spine": "Spine", "femur": "Thigh bone (femur)", "no_plane": "No clear standard view"}
    return names.get(value, value.replace("_", " ").title())


@st.cache_data(show_spinner=False)
def get_config() -> dict:
    return load_config()


@st.cache_resource(show_spinner="Loading the analysis model...")
def get_interpreter(model_path: str):
    return load_interpreter(model_path)


def artifact_status() -> tuple[bool, str]:
    missing = [str(path) for path in (DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH) if not path.exists()]
    return (not missing, "" if not missing else "Missing: " + ", ".join(missing))


def render_prediction(result: dict) -> None:
    predicted = result["predicted_class"]
    confidence = result["confidence"]
    if predicted == "no_plane":
        st.warning("No clear standard ultrasound view was found in this frame.")
        st.write("Try another frame with the anatomy centred, visible, and in focus. This result does not mean that anything is wrong with the pregnancy.")
    else:
        st.success(f"Likely view: {friendly_name(predicted)}")
        st.metric("Model confidence", f"{confidence:.1%}")
        st.caption("Confidence describes how closely the image matches the model's learned examples. It is not a diagnosis or a measure of pregnancy health.")
    chart = pd.DataFrame({"View": [friendly_name(label) for label in result["probabilities"]], "Likelihood": list(result["probabilities"].values())})
    st.write("How the model viewed this frame")
    st.bar_chart(chart.set_index("View"), y="Likelihood", height=260)
    st.caption("The bars show the model's relative likelihood for each supported view; they are not medical probabilities.")


def main() -> None:
    st.markdown("""
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 1rem; background: linear-gradient(135deg, #edf7f6, #f7fbff); border: 1px solid #d7e8e7; margin-bottom: 1.2rem;}
    .hero h1 {margin: 0 0 .35rem 0; color: #123f4a;}
    .hero p {margin: 0; color: #45626a; font-size: 1.05rem;}
    .upload-guide {padding: 1.1rem 1.25rem; border: 1px solid #9ccbc5; border-radius: .8rem; background: #f3fbfa; color: #18343b; margin: 1rem 0 1.25rem 0;}
    .upload-guide h3 {color: #123f4a; margin: 0 0 .75rem 0;}
    .step {padding: .8rem 1rem; border-left: 4px solid #2a9d8f; background: #ffffff; color: #18343b; margin: .35rem 0;}
    </style>
    <div class="hero"><h1>🩺 Taswira Ultrasound Assistant</h1><p>Upload ultrasound images to identify the type of standard view shown.</p></div>
    """, unsafe_allow_html=True)
    st.info("This is a decision-support demonstration. It does not diagnose, assess the baby's health, replace a qualified sonographer, or make referral decisions.")
    st.markdown("""
    <section class="upload-guide">
      <h3>Before you upload: what do I need?</h3>
      <div class="step"><b>1. Use an ultrasound image</b><br>Upload a clear frame, not a document, spreadsheet, video, or ZIP file.</div>
      <div class="step"><b>2. Supported formats</b><br>PNG, JPG/JPEG, BMP, TIF/TIFF. The image should show one ultrasound view with the anatomy visible.</div>
      <div class="step"><b>3. Choose the right option</b><br>Use <b>One frame</b> for a single image, or <b>Multiple frames</b> when you have several images from one sweep.</div>
    </section>
    """, unsafe_allow_html=True)

    ready, status = artifact_status()
    with st.sidebar:
        st.header("About this tool")
        st.write("The model looks for six categories of ultrasound view:")
        st.write("• Head\n• Abdomen\n• Heart\n• Spine\n• Thigh bone\n• No clear standard view")
        st.divider()
        st.subheader("System status")
        if ready:
            st.success("Ready for analysis")
        else:
            st.error("The analysis model is not available.")
            with st.expander("Technical details"):
                st.code(status)
        st.caption("For technical users")
        with st.expander("Model details"):
            st.write("The image is cropped, converted to grayscale, adjusted for contrast, padded to a square, and resized before analysis.")
            st.code("outputs/taswira_plane_finder.tflite\noutputs/pipeline_config.json")
    if not ready:
        st.stop()
    try:
        config = get_config()
        interpreter = get_interpreter(str(DEFAULT_MODEL_PATH))
    except Exception as exc:
        st.error(f"The analysis model could not be loaded: {exc}")
        st.stop()

    single_tab, sweep_tab, info_tab, breech_tab, anomaly_tab = st.tabs(["One frame", "Multiple frames", "How it works", "Breech screening", "Brain screening"])
    with single_tab:
        st.subheader("Check one ultrasound frame")
        st.write("Choose one image. The result tells you which standard view the model thinks it sees.")
        upload = st.file_uploader("Choose an ultrasound image", type=IMAGE_TYPES, key="single", help="Accepted: PNG, JPG/JPEG, BMP, TIF/TIFF")
        if upload:
            try:
                image = Image.open(upload).convert("RGB")
                left, right = st.columns([1, 1])
                with left:
                    st.image(image, caption=upload.name, use_container_width=True)
                with right:
                    with st.spinner("Looking at the image..."):
                        result = predict_frame(interpreter, image, config)
                    render_prediction(result)
            except Exception as exc:
                st.error(f"We could not analyze this image: {exc}")

    with sweep_tab:
        st.subheader("Check several frames from one sweep")
        st.write("Upload the frames in a sweep. The app checks each frame and identifies the strongest supported view found across them.")
        uploads = st.file_uploader("Choose ultrasound images", type=IMAGE_TYPES, accept_multiple_files=True, key="sweep", help="Select several PNG, JPG/JPEG, BMP, TIF/TIFF images from the same scan sweep.")
        if uploads:
            try:
                images = [Image.open(item).convert("RGB") for item in uploads]
                with st.spinner(f"Looking at {len(images)} images..."):
                    results = predict_sweep(interpreter, images, config)
                summary = sweep_summary(results, config)
                if summary["usable"]:
                    st.success("The sweep contains at least one clear standard view.")
                    st.caption("This means the model found a usable-looking frame; it does not confirm that the scan is medically complete.")
                else:
                    st.warning("The sweep did not contain a clearly recognisable standard view.")
                    st.caption("Consider repeating the sweep or asking a qualified sonographer to review it.")
                st.metric("Images checked", summary["frames"])
                best = best_frames(results, config)
                st.subheader("Strongest match for each view")
                cols = st.columns(len(config["classes"]))
                for column, class_name in zip(cols, config["classes"]):
                    row = best[class_name]
                    with column:
                        st.metric(friendly_name(class_name), f"{row['probabilities'][class_name]:.1%}", f"image {row['frame']}")
                table = pd.DataFrame([{ "Image": row["frame"], "Suggested view": friendly_name(row["predicted_class"]), "Model confidence": f"{row['confidence']:.1%}" } for row in results])
                st.dataframe(table, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"We could not analyze this sweep: {exc}")

    with info_tab:
        st.subheader("How the analysis works")
        st.markdown("""
        1. **You provide one or more ultrasound images.**
        2. **The app prepares each image** by cropping the configured area, improving contrast, and resizing it consistently.
        3. **The model compares the image with learned examples** of head, abdomen, heart, spine, thigh-bone, and unclear views.
        4. **You receive a suggested view and confidence score.** For multiple images, the app also reports whether at least one frame looks usable.

        A high score means the image resembles the model's training examples. It does **not** mean the pregnancy is normal, and a low score does **not** diagnose a problem.
        """)
        with st.expander("Technical model information"):
            st.json({"model": "TensorFlow Lite", "backbone": config.get("backbone"), "head": config.get("head"), "image_size": config["img_size"], "classes": config["classes"], "frame_threshold": config["frame_threshold"], "sweep_threshold": config["sweep_threshold"]})

    with breech_tab:
        st.subheader("Breech screening")
        st.warning("This check is not available in the current app.")
        st.write("This deployment can identify standard ultrasound views only. It does not determine whether the baby is breech.")
    with anomaly_tab:
        st.subheader("Brain screening")
        st.warning("This check is not available in the current app.")
        st.write("No brain abnormality result is generated here. The current deployment only identifies the type of ultrasound view.")


if __name__ == "__main__":
    main()
