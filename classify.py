import cv2
import numpy as np
import mediapipe as mp
import joblib

# ── CONFIG ─────────────────────────────────────────────────────────────
MODEL_PATH = "pose_clf.pkl"
# LANDMARKS must match collect.py count
LANDMARKS = [
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
]

# ◀ ADDED: define the same connections used in collect.py
CONNECTIONS = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,  mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW,     mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,    mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,  mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,       mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE,      mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,      mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE,     mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]

# ── HELPER FUNCTIONS (as in train.py) ─────────────────────────────────
def normalize_skeleton(skeleton):
    hip_left, hip_right = skeleton[6], skeleton[7]
    center = (hip_left + hip_right) / 2
    coords = skeleton - center
    shoulder_left, shoulder_right = skeleton[0], skeleton[1]
    torso_length = np.linalg.norm(((shoulder_left + shoulder_right)/2) - center)
    coords /= (torso_length + 1e-6)
    return coords

def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    v1 = a - b; v2 = c - b
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def extract_geom_features(skeleton):
    dist_we = distance(skeleton[4], skeleton[2])
    elbow_ang = angle(skeleton[0], skeleton[2], skeleton[4])
    return [dist_we, elbow_ang]

def temporal_deltas(prev, curr):
    return (curr - prev).flatten()

# ── SETUP mediapipe & model ─────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    # +smoothness | -precision
    # model_complexity=0,                # lite model for speed
    # smooth_landmarks=False,            # disable smoothing to cut latency
    # enable_segmentation=False,         # skip segmentation mask
)
clf = joblib.load(MODEL_PATH)

# ── REAL-TIME LOOP WITH FEATURE ENGINEERING ─────────────────────────────
cap = cv2.VideoCapture(0)

# ◀ ADDED: create a resizable window and enlarge it 1.2×
window_name = "Pose Classification"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)                # allow resizing
# Will resize after first frame is read

print("[classify] Starting webcam. Press ESC to quit.")

prev_norm = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ◀ ADDED: on the first frame, enlarge the window
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == 1:
        h, w = frame.shape[:2]
        cv2.resizeWindow(window_name, int(w * 1.2), int(h * 1.2))

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # 1) Extract raw skeleton (n_landmarks × 2)
        h, w = img.shape[:2]
        sk = []
        for lm_enum in LANDMARKS:
            lm = results.pose_landmarks.landmark[lm_enum.value]
            sk.append([lm.x, lm.y])
        sk = np.array(sk)

        # 2) Normalize coords
        norm = normalize_skeleton(sk)

        # 3) Build feature vector
        flat = norm.flatten()
        geom = extract_geom_features(norm)
        if prev_norm is None:
            delta = np.zeros_like(flat)
        else:
            delta = temporal_deltas(prev_norm, norm)
        feat = np.hstack([flat, geom, delta]).reshape(1, -1)
        prev_norm = norm

        # 4) Predict
        pred  = clf.predict(feat)[0]
        proba = clf.predict_proba(feat)[0].max()

        # ◀ ADDED: draw the skeleton on the image
        for lm_enum in LANDMARKS:
            lm = results.pose_landmarks.landmark[lm_enum.value]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)            # yellow joints

        for (start, end) in CONNECTIONS:
            ls = results.pose_landmarks.landmark[start.value]
            le = results.pose_landmarks.landmark[end.value]
            xs, ys = int(ls.x * w), int(ls.y * h)
            xe, ye = int(le.x * w), int(le.y * h)
            cv2.line(img, (xs, ys), (xe, ye), (255, 255, 0), 2)       # cyan bones

        # Overlay result
        text = f"{pred} ({proba:.2f})"
        # ◀ MODIFIED: make font color red, and increase font scale to 1.2
        cv2.putText(
            img,
            text,
            (10, 40),                               # lower on frame to avoid overlap
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,                                    # bigger text
            (0, 0, 255),                            # red color in BGR
            3                                       # thicker stroke
        )

    cv2.imshow(window_name, img)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
