import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.python.solutions.pose_connections import POSE_CONNECTIONS
from matplotlib.patches import Circle

# ── CONFIG ───────────────────────────────────────────────────────────────
CSV_FILE   = "data.csv"
OUTPUT_DIR = "pose_images"
WIDTH, HEIGHT = 640, 480    # output canvas size

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)

# ── DEFINE FULL-BODY LANDMARKS ───────────────────────────────────────────
ALL_LMS = list(mp.solutions.pose.PoseLandmark)
# drop face landmarks 1–10 (keep 0 = NOSE), and drop hand/finger 17–22
EXCLUDE = set(range(1, 11)) | set(range(17, 23))
CANDIDATE_LMS = [lm for lm in ALL_LMS if lm.value not in EXCLUDE]

# ── KEEP ONLY JOINTS PRESENT IN CSV ───────────────────────────────────────
cols = set(df.columns)
landmarks = [
    lm for lm in CANDIDATE_LMS
    if f"{lm.name.lower()}_x" in cols and f"{lm.name.lower()}_y" in cols
]

# Map landmark.name → its index in our filtered landmarks list
name_to_idx = {lm.name.lower(): idx for idx, lm in enumerate(landmarks)}
lm_index   = {lm.value: idx for idx, lm in enumerate(landmarks)}

# ── BUILD SKELETON CONNECTIONS ──────────────────────────────────────────
connections = []
for start, end in POSE_CONNECTIONS:
    s_val = start.value if hasattr(start, "value") else start
    e_val = end.value   if hasattr(end,   "value") else end
    if s_val in lm_index and e_val in lm_index:
        connections.append((lm_index[s_val], lm_index[e_val]))

# ── SAMPLE ONE RANDOM POSE PER LABEL ────────────────────────────────────
sampled = df.groupby("label", group_keys=False).sample(n=1, random_state=42)

# ── DRAW & SAVE ─────────────────────────────────────────────────────────
for _, row in sampled.iterrows():
    label = row["label"]

    # 1) extract & scale all joint coords
    xs = [row[f"{lm.name.lower()}_x"] * WIDTH  for lm in landmarks]
    ys = [row[f"{lm.name.lower()}_y"] * HEIGHT for lm in landmarks]

    # 2) compute head center & radius
    # shoulder positions must exist
    ls_x = row["left_shoulder_x"]  * WIDTH
    ls_y = row["left_shoulder_y"]  * HEIGHT
    rs_x = row["right_shoulder_x"] * WIDTH
    rs_y = row["right_shoulder_y"] * HEIGHT

    shoulder_mid_x = (ls_x + rs_x) / 2
    shoulder_mid_y = (ls_y + rs_y) / 2
    shoulder_dist  = math.hypot(ls_x - rs_x, ls_y - rs_y)
    head_radius    = shoulder_dist / 2

    if "nose" in name_to_idx:
        # use actual nose if available
        center_x = row["nose_x"] * WIDTH
        center_y = row["nose_y"] * HEIGHT
    else:
        # fallback: center above shoulders
        center_x = shoulder_mid_x
        center_y = shoulder_mid_y - head_radius

    # 3) plot stick–figure
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    # bones
    for s, e in connections:
        ax.plot([xs[s], xs[e]], [ys[s], ys[e]], color="black", linewidth=2)
    # joints
    ax.scatter(xs, ys, s=30, color="red", zorder=3)
    # head circle
    circ = Circle((center_x, center_y), head_radius,
                  fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(circ)

    # 4) finalize & save
    ax.invert_yaxis()   # match MediaPipe’s origin
    ax.axis("off")
    out_path = os.path.join(OUTPUT_DIR, f"{label}.png")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
