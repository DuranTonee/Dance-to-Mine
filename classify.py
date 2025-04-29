import cv2
import numpy as np
import mediapipe as mp
import joblib

# ── CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH = "pose_clf.pkl"

# same joints as in collect.py
LANDMARKS = [
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
]

# ── SETUP mediapipe & model ───────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
clf = joblib.load(MODEL_PATH)

# ── REAL-TIME LOOP ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("[classify] Starting webcam. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        # build feature vector
        feat = []
        for lm_enum in LANDMARKS:
            lm = results.pose_landmarks.landmark[lm_enum.value]
            feat += [lm.x, lm.y]

        # predict
        pred    = clf.predict([feat])[0]
        proba   = clf.predict_proba([feat])[0].max()

        # overlay
        text = f"{pred} ({proba:.2f})"
        cv2.putText(img, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Pose Classification", img)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
