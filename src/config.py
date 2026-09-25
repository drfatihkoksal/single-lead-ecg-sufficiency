"""Central configuration: paths, signal geometry, label space, SNOMED CT mapping, lead sets.

Study-1 layout (Kaggle Chapman release) lives in artifacts/; every resubmission run lives
in artifacts_v2/<cohort>/<arch>/seed<k>/ (see ds_paths).
"""
import os
from pathlib import Path

# ---- paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "chapman_siaoxing_with_extracted_features.parquet"
ART = ROOT / "artifacts"
CACHE = ART / "cache"
MODELS = ART / "models"
METRICS = ART / "metrics"
SPLITS = ART / "splits"
FIGURES = ROOT / "figures"
# v2 (resubmission) outputs live apart from the study-1 artifacts, which the
# noise study loads as frozen models and must never be overwritten
ART_V2 = ROOT / "artifacts_v2"

# ---- signal geometry ------------------------------------------------------
FS = 500              # Hz
N_SAMPLES = 5000      # 10 s
N_LEADS = 12
# standard 12-lead order (matches lead-major block order in the parquet)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ---- label space (11 classes, parquet column name -> short code) ----------
LABEL_COLS = [
    "Atrial Fibrillation",
    "Atrial Flutter",
    "T wave Change",
    "Axis left shift",
    "Sinus Bradycardia",
    "Supraventricular Tachycardia",
    "Sinus Rhythm",
    "Sinus Tachycardia",
    "ST-T Change",
    "Left_Ventricular_Hypertrophy",
    "RBBB",
]
CLASS_CODE = {
    "Atrial Fibrillation": "AF",
    "Atrial Flutter": "AFL",
    "T wave Change": "TWC",
    "Axis left shift": "LAD",
    "Sinus Bradycardia": "SB",
    "Supraventricular Tachycardia": "SVT",
    "Sinus Rhythm": "SR",
    "Sinus Tachycardia": "ST",
    "ST-T Change": "STTC",
    "Left_Ventricular_Hypertrophy": "LVH",
    "RBBB": "RBBB",
}
CLASSES = [CLASS_CODE[c] for c in LABEL_COLS]   # short codes, aligned to LABEL_COLS order
N_CLASSES = len(CLASSES)

# ---- split / repro --------------------------------------------------------
SEED = 1337
SPLIT_FRACS = (0.70, 0.15, 0.15)   # train / val / test, record-level (Chapman ~ 1 ECG/patient)

# ---- training defaults ----------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8           # early stop on val macro-AUPRC


def lead_index(name: str) -> int:
    return LEADS.index(name)


# ---- multi-dataset path routing -------------------------------------------
# Chapman keeps the original flat artifact dirs; external sets go under
# artifacts/<dataset>/.  Same 11-class label space/order is used everywhere.
class _Paths:
    def __init__(self, cache, splits, models, metrics):
        self.cache, self.splits, self.models, self.metrics = cache, splits, models, metrics


# cohorts whose signal cache/split already exists in the study-1 layout; v2 runs
# reuse them read-only instead of duplicating several GB of float32 signals
# (PTB-XL v2 has its own narrow labels; its signals.npy is a symlink to the legacy one)
# v2 "chapman" is the raw PhysioNet release; the Kaggle release is kept as "chapman_kaggle"
_LEGACY_DATA = {"chapman_kaggle": (CACHE, SPLITS)}
COHORTS = ["chapman", "ningbo", "ptbxl_snomed", "georgia"]


def ds_paths(dataset: str = "chapman", arch: str = None, seed: int = None) -> "_Paths":
    """Without arch: the study-1 layout.  With arch: artifacts_v2/<ds>/<arch>/seed<k>/."""
    if arch is None:
        if dataset == "chapman":
            return _Paths(CACHE, SPLITS, MODELS, METRICS)
        base = ART / dataset
        return _Paths(base / "cache", base / "splits", base / "models", base / "metrics")
    cache, splits = _LEGACY_DATA.get(dataset, (ART_V2 / dataset / "cache", ART_V2 / dataset / "splits"))
    run = ART_V2 / dataset / arch / f"seed{SEED if seed is None else seed}"
    return _Paths(cache, splits, run / "models", run / "metrics")


# ---- lead configurations ---------------------------------------------------
# a lead spec is a single lead ("V1"), "ALL", a named set, or "+"-joined leads
LEAD_SETS = {
    "LIMB6": ["I", "II", "III", "aVR", "aVL", "aVF"],   # six-lead limb devices
    "CH3": ["I", "II", "V2"],                          # Challenge 2021 three-lead
    "CH4": ["I", "II", "III", "V2"],                   # Challenge 2021 four-lead
}
DEVICE_SPECS = ["I", "II", "I+II", "LIMB6", "CH3", "CH4"]


# PTB-XL source
PTBXL_ROOT = Path(os.environ.get("PTBXL_ROOT", ROOT.parent / "mi_localisation" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"))

# PTB-XL label mapping: our class code -> SCP codes (and special axis handling)
PTBXL_SCP_MAP = {
    "AF":   ["AFIB"],
    "AFL":  ["AFLT"],
    "TWC":  ["NDT", "TAB_", "INVT"],
    "LAD":  ["__AXIS_LAD__"],           # from heart_axis field, not scp_codes
    "SB":   ["SBRAD"],
    "SVT":  ["SVTAC", "PSVT"],
    "SR":   ["SR"],
    "ST":   ["STACH"],
    "STTC": ["NST_", "ISC_", "STD_", "STE_", "ISCAL", "ISCIN", "ISCIL",
             "ISCAS", "ISCLA", "ISCAN"],
    "LVH":  ["LVH"],
    "RBBB": ["CRBBB", "IRBBB"],
}
PTBXL_AXIS_LAD = {"LAD", "ALAD"}        # heart_axis values counted as left-axis positive

# Narrow PTB-XL map for the resubmission: one statement per class, matching the
# single-code SNOMED_MAP definitions (the study-1 map above is the broad variant)
PTBXL_SCP_MAP_NARROW = {**PTBXL_SCP_MAP,
    "TWC":  ["NDT", "TAB_"],            # non-specific T abnormality (INVT -> broad)
    "SVT":  ["SVTAC"],                  # (PSVT -> broad)
    "STTC": ["NST_"],                   # non-specific ST change (ST dev./ischaemia -> broad)
    "RBBB": ["CRBBB"],                  # complete only (IRBBB -> broad)
}


# ---- SNOMED-CT label mapping (Georgia, and the raw Chapman/Ningbo release) --
# Primary map: the single code each Chapman class was defined by.  Georgia and the
# PhysioNet ecg-arrhythmia release share this vocabulary (ningbo/ConditionNames_SNOMED-CT.csv).
SNOMED_MAP = {
    "AF":   ["164889003"],
    "AFL":  ["164890007"],
    "TWC":  ["164934002"],            # T-wave change
    "LAD":  ["39732003"],             # axis left shift
    "SB":   ["426177001"],
    "SVT":  ["426761007"],
    "SR":   ["426783006"],
    "ST":   ["427084000"],
    "STTC": ["428750005", "55930002"],   # nonspecific ST-T abnormality; ST changes (Ningbo's
                                         # code, and the target of PTB-XL NST_)
    "LVH":  ["164873001", "55827005"],   # LVH; LV high voltage = the Chapman/Ningbo ECG-LVH label
    "RBBB": ["59118001", "713427006"],   # RBBB, complete RBBB
}
# strict variant: LVH diagnosis code only (Chapman then has 15 positives)
SNOMED_LVH_STRICT = ["164873001"]
# classes a source cannot supply: Ningbo codes every atrial fibrillation/flutter as
# flutter (Challenge 2021 dx table: AF 0, AFL 7615), so AF is empty and AFL means
# "AF or AFL"; both are left out of Ningbo's cross-cohort comparisons
UNAVAILABLE = {"ningbo": ["AF", "AFL"], "ningbo_broad": ["AF", "AFL"]}
# Broad map for the label-definition sensitivity analysis (closer to the PTB-XL map,
# which pools ischaemia codes into STTC and incomplete RBBB into RBBB)
SNOMED_MAP_BROAD = {**SNOMED_MAP,
    "TWC":  ["164934002", "59931005"],                          # + T-wave inversion
    "SVT":  ["426761007", "713422000", "67198005"],             # + atrial tach., PSVT
    "STTC": ["428750005", "55930002", "429622005", "164930006", "164931005",  # + ST dep./elev.
             "425623009", "425419005", "426434006"],            # + lateral/inferior/anterior ischaemia
    "RBBB": ["59118001", "713427006", "713426002"],             # + incomplete RBBB
}


# ---- interpretation flags (Table 3, Fig. 2) ---------------------------------
# dagger: twelve-lead AUPRC below 0.50 or fewer than MIN_POS_TEST positive test recordings
# double dagger: label inconsistent with the signal (artifacts_v2/review/label_*.csv): in PTB-XL
# only 15% of regular recordings with a heart rate below 60/min carry the sinus bradycardia code;
# in Georgia sinus rhythm is never coded together with a morphological diagnosis
MIN_POS_TEST = 30
LABEL_INCONSISTENT = {("ptbxl_snomed", "SB"), ("georgia", "SR")}


def flag(cohort, cls, low_ceiling, n_pos):
    """Interpretation flag for a class in a cohort: '†', '‡' or '' (none where the class is not coded)."""
    if cls in UNAVAILABLE.get(cohort, []):
        return ""
    if (cohort, cls) in LABEL_INCONSISTENT:
        return "‡"
    return "†" if (bool(low_ceiling) or n_pos < MIN_POS_TEST) else ""
