# Streamlit deployment

The deployment dashboard is implemented in `app.py`. It uses the packaged TensorFlow Lite model and matching configuration:

- `outputs/taswira_plane_finder.tflite`
- `outputs/pipeline_config.json`

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The app supports single-frame inference and multi-frame sweep analysis. It displays the six plane classes, confidence probabilities, best frames, and sweep usability.

Breech and brain-anomaly tabs are intentionally marked unavailable. Those analyses currently exist only inside the notebook and have not been exported with standalone model artifacts and preprocessing metadata.

This is a decision-support demonstration. It does not diagnose, replace a qualified sonographer, or make autonomous referral decisions.
