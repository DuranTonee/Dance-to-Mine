import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

# ── FEATURE ENGINEERING UTILITIES ────────────────────────────────────

def normalize_skeleton(skeleton):
    """
    Translation/Scale Normalization:
    1. Center at hip midpoint,  2. Scale by torso length.
    skeleton: np.array shape (n_landmarks, 2)
    """
    # Hip midpoint (assuming hips are at indices 6 & 7 in LANDMARKS)
    hip_left, hip_right = skeleton[6], skeleton[7]
    center = (hip_left + hip_right) / 2
    coords = skeleton - center

    # Torso length: shoulder midpoint to hip midpoint
    shoulder_left, shoulder_right = skeleton[0], skeleton[1]
    torso_length = np.linalg.norm(((shoulder_left + shoulder_right)/2) - center)
    coords /= (torso_length + 1e-6)
    return coords

def distance(a, b):
    """Euclidean distance between two 2D points."""
    return np.linalg.norm(a - b)

def angle(a, b, c):
    """
    Angle at point b between points a–b–c.
    Returns angle in degrees.
    """
    v1 = a - b
    v2 = c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def extract_geom_features(skeleton):
    """
    Relative Distances & Angles:
    Example: wrist(4), elbow(2), shoulder(0) indices in normalized skeleton.
    Returns list [dist_wrist_elbow, elbow_angle].
    """
    dist_we = distance(skeleton[4], skeleton[2])
    elbow_ang = angle(skeleton[0], skeleton[2], skeleton[4])
    return [dist_we, elbow_ang]

def temporal_deltas(prev, curr):
    """
    Temporal Delta Features:
    Differences between consecutive skeleton frames.
    """
    deltas = curr - prev
    return deltas.flatten()

# ── LOAD DATA & INITIAL PROCESSING ───────────────────────────────────
df = pd.read_csv("data.csv")
y = df["label"].values
# Raw landmarks: shape (n_samples, n_landmarks*2)
raw = df.drop("label", axis=1).values
n_landmarks = raw.shape[1] // 2
# Reshape to (n_samples, n_landmarks, 2)
skeletons = raw.reshape(-1, n_landmarks, 2)

# Prepare feature matrix
features = []
prev_norm = None

for sk in skeletons:
    # 1) Normalize skeleton
    norm = normalize_skeleton(sk)
    # 2) Flatten normalized coords
    flat = norm.flatten()

    # 3) Geometric features
    geom = extract_geom_features(norm)

    # 4) Temporal delta (zeros for first frame)
    if prev_norm is None:
        delta = np.zeros_like(flat)
    else:
        delta = temporal_deltas(prev_norm, norm)
    prev_norm = norm

    # Combine all features
    features.append(np.hstack([flat, geom, delta]))

X = np.array(features)

# ── STRATIFIED CROSS-VALIDATION ───────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Pipeline: scale → RandomForest
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )
    clf.fit(X_tr, y_tr)
    score = clf.score(X_te, y_te)
    print(f"[train] Fold {fold} accuracy: {score*100:.1f}%")
    fold_scores.append(score)

print(f"[train] Mean CV accuracy: {np.mean(fold_scores)*100:.1f}%")

# ── FINAL TRAIN & SAVE MODEL ─────────────────────────────────────────
print("[train] Fitting final model on all data…")
final_clf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
)
final_clf.fit(X, y)
joblib.dump(final_clf, "pose_clf.pkl")
print("[train] Saved classifier to 'pose_clf.pkl'")
