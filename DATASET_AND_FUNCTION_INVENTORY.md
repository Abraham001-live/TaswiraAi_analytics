# Taswira AI Analytics: Dataset and Function Inventory

Source inspected: `taswira_final_short_results.ipynb` and the project folder.

This inventory separates raw datasets from generated caches, model files, and result tables. The notebook is the main executable source; it does not contain separate application `.py` files outside the virtual environments.

## 1. Raw datasets currently present

### 1.1 NatalIA PBFUS1 — primary plane-finder dataset

Location: `NatalIA PBFUS1/code` and `NatalIA PBFUS1.zip`

Files and contents:

- `resume.csv`: 19,407 frame-level rows with columns `file_name`, `studie`, `class`, `value`, and `image`.
- `metadata.csv`: 90 sweep-level rows with study name, sweep protocol, fetal position, and volunteer metadata.
- 90 examination folders containing JPEG ultrasound frames.
- Current extracted image count: 19,407 JPEG frames.

The notebook converts the frame labels into six classes:

| ID | Class | Meaning |
|---:|---|---|
| 0 | `biparietal` | fetal brain / head view |
| 1 | `abdominal` | fetal abdomen view |
| 2 | `heart` | fetal thorax / heart view |
| 3 | `spine` | fetal spine view |
| 4 | `femur` | fetal femur view |
| 5 | `no_plane` | no standard target plane |

The current raw label distribution in `resume.csv` is:

| Label value | Frames |
|---:|---:|
| 0 | 42 |
| 1 | 63 |
| 2 | 61 |
| 3 | 134 |
| 4 | 46 |
| 5 | 19,061 |

The notebook also uses the metadata fields as follows:

- `protocol` becomes `sweep_type`: horizontal, vertical, diagonal 1, or diagonal 2.
- `position` becomes `fetal_position`: OA, OP, SA, or SP.
- OA/OP are treated as head-down; SA/SP are treated as breech for the exploratory breech model.

Example loading code:

```python
DATA_DIR = NATALIA_META.parent
frames = pd.read_csv(DATA_DIR / "resume.csv")
studies = pd.read_csv(DATA_DIR / "metadata.csv")
```

### 1.2 Fetal Brain Abnormalities Ultrasound — exploratory anomaly dataset

Location: `Fetal Brain Abnormalities Ultrasound.v1i.multiclass/` and `Fetal Brain Abnormalities Ultrasound.v1i.multiclass.zip`

This is a Roboflow classification export. The local README states that it contains 1,768 images, while the current extracted directory contains:

- `train/`: 1,419 images
- `valid/`: 175 images
- `test/`: 177 images
- Total extracted image files: 1,771

The notebook treats this dataset as a study-only, exploratory brain-anomaly demonstration. It groups Roboflow augmented copies by their original filename pattern, drops classes with fewer than 10 images, and performs grouped cross-validation.

The local filenames encode classes such as:

- normal
- mild, moderate, and severe ventriculomegaly
- arachnoid cyst
- cerebellar hypoplasia
- colpocephaly
- encephalocele
- hydranencephaly
- intracranial haemorrhage
- intracranial tumour
- mega cisterna magna
- porencephaly
- holoprosencephaly
- vein of Galen malformation
- Arnold-Chiari malformation

The dataset is not used to make a clinical diagnosis. The notebook explicitly produces referral-oriented, non-diagnostic messages and includes an unfamiliar-image guard.

Example loading behavior:

```python
ANOM_ROOT = pathlib.Path("Fetal Brain Abnormalities Ultrasound.v1i.multiclass")
anom_images = list(ANOM_ROOT.rglob("*.jpg"))
```

## 2. Datasets referenced by the notebook but not currently present

The notebook supports these external datasets, but their metadata files were not found in the project folder during inspection:

### Spain FETAL_PLANES_DB

Expected metadata file: `FETAL_PLANES_DB_data.csv`

Expected archive pattern: a ZIP filename containing `fetal_planes`.

Expected mapped classes:

- fetal brain, trans-thalamic only → `biparietal`
- fetal abdomen → `abdominal`
- fetal thorax → `heart`
- fetal femur → `femur`
- other or maternal cervix → `no_plane`

When present, the notebook randomly selects up to 400 images per mapped class for the `Dummy + Spain` model.

### Africa African planes database

Expected metadata file: `African_planes_database.csv`

Expected archive pattern: a `.tar.xz` archive.

Expected mapped classes:

- brain → `biparietal`
- abdomen → `abdominal`
- thorax → `heart`
- femur → `femur`

The notebook expects 450 African images and records country/center and patient identifiers for patient-level bootstrap evaluation. The supported country list is Algeria, Egypt, Ghana, Malawi, and Uganda.

Because these datasets are absent, the Spain and Africa branches are conditionally skipped when the notebook runs.

## 3. Derived datasets and artifacts

These are not independent raw datasets; they are generated from the raw data or are model/evaluation outputs.

### `cache/`

- `frames.npy`: 224×224 preprocessed NatalIA grayscale frames stored as a memory-mapped array.
- `frame_stats.csv`: brightness, contrast, and perceptual-difference hash for each NatalIA frame.
- `frames_key.json`: preprocessing configuration and frame-count key.
- `feat_efficientnetb0_*.npy` and `feat_mobilenetv2_*.npy`: frozen-image feature vectors for NatalIA.
- `ext_*_*.npy`: cached feature vectors for optional external datasets and the anomaly dataset.
- `backbone_*_224.weights.h5`: cached ImageNet backbone weights.

### `outputs/`

- `phantom_results.csv`: plane-finder evaluation metrics.
- `headline_results.csv`: summary metrics and targets.
- `cv_results_by_config.csv`: cross-validation results by configuration.
- `pipeline_config.json`: saved configuration.
- `taswira_plane_finder.keras`: packaged Keras plane-finder model.
- `taswira_plane_finder.tflite`: TensorFlow Lite deployment model.

Other model artifacts in the project root, including `natalia_plane_model.keras` and `natalia_plane_model.tflite`, are trained/exported model files rather than datasets.

## 4. Function inventory

All functions below are defined in `taswira_final_short_results.ipynb`.

### Dataset discovery and loading

#### `search(pattern)`

Searches below the notebook directory while excluding `cache` and `outputs`.

```python
hits = search("resume.csv")
```

Returns a sorted list of matching `Path` objects.

#### `ensure(meta_name, archives, target)`

Looks for a metadata file and extracts a candidate ZIP or TAR archive if the metadata is missing.

```python
NATALIA_META = ensure(
    "resume.csv",
    [Path("NatalIA PBFUS1.zip")],
    Path("NatalIA PBFUS1")
)
```

Returns the first metadata path found, or `None`.

#### `resolve_index(root)`

Builds a case-insensitive index of image filenames and stems for external datasets.

```python
image_index = resolve_index(Path("FETAL_PLANES_DB"))
image_path = image_index.get("image_001.jpg")
```

Returns a dictionary mapping filename/stem to `Path`.

#### `map_view(plane, brain="")`

Maps external dataset labels to the notebook's common plane classes.

```python
map_view("Fetal thorax")                 # "heart"
map_view("Fetal brain", "trans-thalamic") # "biparietal"
map_view("Other")                        # "no_plane"
```

Returns a class name or `None` for an unsupported view.

### Image preprocessing

#### `stretch_pad(a)`

Contrast-stretches a grayscale array using the 1st and 99.5th percentiles, pads it to a square, and resizes it to 224×224.

```python
image_224 = stretch_pad(gray_array)
```

Returns a `uint8` NumPy array with shape `(224, 224)`.

#### `preprocess(img_gray)`

Applies the fixed NatalIA ultrasound-sector crop followed by `stretch_pad`.

```python
processed = preprocess(Image.open(frame_path).convert("L"))
```

#### `preprocess_external(img_gray)`

Finds a non-background crop independently for Spain, Africa, or anomaly images, then applies the same contrast, padding, and resizing pipeline.

```python
processed = preprocess_external(Image.open(external_path).convert("L"))
```

### Feature extraction

#### `make_backbone(name)`

Creates an ImageNet-backed EfficientNetB0 or MobileNetV2 feature extractor.

```python
backbone = make_backbone("efficientnetb0")
```

#### `get_backbone(name)`

Creates a backbone and loads or saves its cached weights under `cache/`.

```python
backbone = get_backbone("mobilenetv2")
```

#### `embed(name, load_batch, n, path)`

Runs a backbone over batches of images, reusing a cached `.npy` feature file when possible.

```python
features_array = embed("efficientnetb0", load_batch, len(paths), cache_path)
```

#### `features(df_paths=None, tag=None)`

Extracts and concatenates EfficientNetB0 and MobileNetV2 features. With no paths, it uses cached NatalIA frames; with paths, it preprocesses external images.

```python
X = features()                         # NatalIA features
X_anomaly = features(anom["path"].tolist(), "brainanom")
```

### Plane model training and prediction

#### `fit_logreg(Xt, yt)`

Standardizes feature vectors and trains a balanced logistic-regression classifier.

```python
scaler, classifier = fit_logreg(X_train, y_train)
```

Returns `(scaler, classifier)`.

#### `predict(model, Xq)`

Produces six-column plane probabilities, inserting zero columns for classes absent from a particular training split.

```python
P = predict((scaler, classifier), X_test)
```

Returns an array shaped `(number_of_images, 6)`.

#### `train(idx, rng, use_spain)`

Trains the plane finder on selected NatalIA indices, samples no-plane negatives, optionally adds Spain images, and performs hard-negative mining.

```python
model = train(
    np.arange(len(frames)),
    np.random.default_rng(SEED),
    use_spain=False
)
```

#### `decide(P, tau)`

Converts probabilities into a plane class when the best plane probability exceeds `tau`; otherwise returns `no_plane`.

```python
predicted_labels = decide(P, tau=0.50)
```

#### `plane_f1(y, pred)`

Computes macro-F1 over detected plane classes while excluding `no_plane`.

```python
score = plane_f1(y_true, predicted_labels)
```

#### `sweep_scores(P, idx)`

Aggregates frame predictions into one row per ultrasound sweep, recording the maximum plane score and whether a true plane exists.

```python
sweep_table = sweep_scores(P, frame_indices)
```

#### `tune(P, idx)`

Selects a frame threshold using plane macro-F1 and a sweep threshold targeting 90% sensitivity on plane-containing sweeps.

```python
thresholds = tune(P_validation, validation_indices)
# {'tau': ..., 'tau_sweep': ...}
```

#### `grouped_folds(df, k, seed=SEED)`

Creates stratified folds while keeping every frame from the same video/sweep in one fold.

```python
fold_ids = grouped_folds(frames, 5)
```

#### `cross_validate(use_spain)`

Runs five-fold out-of-fold evaluation, including inner threshold tuning and optional Spain training data.

```python
cv_result = cross_validate(use_spain=False)
```

Returns out-of-fold probabilities, predictions, and per-fold thresholds.

### Evaluation and deployment helpers

#### `boot_ci(stat)`

Bootstraps metrics by resampling complete sweeps rather than individual frames.

```python
low, high = boot_ci(lambda indices: plane_f1(Y[indices], predictions[indices]))
```

#### `hit_at(P, k=5)`

Checks whether one of the top-k frames contains each plane present in a sweep.

```python
top_five_hit_rate = hit_at(P, k=5)
```

#### `run(x)`

Runs one preprocessed frame through the TensorFlow Lite deployment interpreter.

```python
probabilities = run(x_chk[0])
```

Returns the TFLite output probability vector.

### Breech-position analysis

#### `boot_sweeps(stat)`

Bootstraps the breech metric over complete sweeps.

```python
accuracy_ci = boot_sweeps(lambda i: (pred_b[i] == yb[i]).mean())
```

#### `auc_or_nan(y, score)`

Computes ROC-AUC and returns `NaN` when the supplied labels do not contain both classes.

```python
auc = auc_or_nan(yb, 1 - head_best)
```

### Brain anomaly screening

#### `boot_groups(stat)`

Bootstraps anomaly metrics by original image group, preventing Roboflow augmented copies from being treated as independent originals.

```python
ci = boot_groups(lambda i: f1_score(ya[i], pred_a[i], average="macro"))
```

#### `nice(c)`

A small lambda helper that turns dataset class names into readable display names and corrects known spelling variants.

```python
nice("m-magna")  # "mega cisterna magna"
```

#### `brain_message(P)`

Converts anomaly probabilities into a safe status tier and non-diagnostic message: `refer`, `consult`, or `no_flag`.

```python
tier, probability, message = brain_message(probability_vector)
```

#### `unit(Z)`

Standardizes final anomaly features and L2-normalizes them for similarity/OOD comparison.

```python
normalized = unit(anomaly_features)
```

#### `brain_check(xf)`

Applies the unfamiliar-image guard first, then runs the final anomaly classifier and returns a safe result dictionary.

```python
result = brain_check(image_features)
# {'status': 'no_flag', ...}
```

Possible statuses include `unavailable`, `not_assessed`, `refer`, `consult`, and `no_flag`.

### Midwife visit workflow

#### `visit_data(video)`

Collects the out-of-fold frame predictions for one sweep, selects the best candidate frame for each view, evaluates scan usability, checks breech probability, and optionally performs the brain check on the captured head view.

```python
data = visit_data(video_id)
```

#### `midwife_visit(video, danger_signs=(), midwife="")`

Renders an HTML visit card. It applies the safety order: danger signs first, unusable scan second, exploratory defect flag, breech flag, then routine result.

```python
midwife_visit(
    video_id,
    danger_signs=["Vaginal bleeding"],
    midwife="Demo midwife"
)
```

#### `on_run(_)`

The ipywidgets button callback. It reads the selected sweep, checked danger signs, and midwife name, then calls `midwife_visit`.

```python
btn.on_click(on_run)
```

## 5. Important execution notes

- NatalIA is the only complete primary dataset currently used by the plane-finder pipeline.
- The Spain and Africa branches cannot produce results until their metadata and images are added.
- The brain-anomaly branch is locally available, but it is explicitly exploratory and should not be presented as a validated clinical model.
- The notebook's cached files can make later runs appear to load a dataset even when the raw images are not being reprocessed. The raw source remains `NatalIA PBFUS1` or the anomaly dataset directory.
- `preprocess` is intended for NatalIA frames; `preprocess_external` is intended for images from other datasets.
- `visit_data` and `midwife_visit` depend on earlier notebook state such as `CV`, `base`, `sweep_df`, `VID_IDX`, `X`, and `brain_check`; they are not standalone functions until the notebook has been executed through the preceding sections.
