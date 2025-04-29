import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import make_pipeline
from sklearn.ensemble       import RandomForestClassifier
import joblib

# ── LOAD & SPLIT ─────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")
X = df.drop("label", axis=1)    # features: all landmark coords
y = df["label"]                 # target: your pose names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ── TRAINING PIPELINE ────────────────────────────────────────────────────
# 1) Standardize features → 2) RandomForest
clf = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1         # use all CPU cores
    )
)

print("[train] Fitting model…")
clf.fit(X_train, y_train)

# ── EVALUATION ───────────────────────────────────────────────────────────
acc = clf.score(X_test, y_test)
print(f"[train] Test accuracy: {acc*100:.1f}%")

# ── SAVE MODEL ───────────────────────────────────────────────────────────
joblib.dump(clf, "pose_clf.pkl")
print("[train] Saved classifier to 'pose_clf.pkl'")
