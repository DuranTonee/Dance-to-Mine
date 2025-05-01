# classify_control.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyautogui
import mouse
import time
from collections import deque

# ── CONFIG ─────────────────────────────────────────────────────────────
MODEL_PATH       = "pose_clf.pkl"
IGNORE_THRESHOLD = 0.2    # seconds to ignore very short detections
MOUSE_SPEED      = 400    # px per second (was 500)
CLICK_HOLD_TIME  = 1.5    # seconds (was 1.0)

# exactly 12 landmarks
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

CONNECTIONS = [
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW,    mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,    mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,  mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,       mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE,      mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,      mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE,     mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]

# ── FEATURE HELPERS ────────────────────────────────────────────────────
def normalize_skeleton(skel):
    hip_l, hip_r = skel[6], skel[7]
    center = (hip_l + hip_r) / 2
    coords = skel - center
    sh_l, sh_r = skel[0], skel[1]
    torso = np.linalg.norm(((sh_l + sh_r)/2) - center) + 1e-6
    return coords / torso

def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    v1, v2 = a - b, c - b
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def extract_geom(skel):
    return [
        distance(skel[4], skel[2]),        # wrist–elbow
        angle(skel[0], skel[2], skel[4])   # shoulder–elbow–wrist angle
    ]

def temporal_delta(prev, curr):
    return (curr - prev).flatten()

# ── CONTROL HELPERS ────────────────────────────────────────────────────
def pan_mouse(speed, dt):
    # move at `speed` px/sec for this frame
    dx = speed * dt
    mouse.move(dx, 0, absolute=False)

# ── SETUP MODEL & STATE ────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
clf = joblib.load(MODEL_PATH)
pyautogui.FAILSAFE = True

# Debounce & action state
last_frame_pred   = None
frame_pred_start  = 0.0
stable_pred       = None
prev_stable_pred  = None
prev_norm         = None
last_time         = time.time()
w_down            = False
click_down        = False
click_start       = 0.0
updown_hist       = deque(maxlen=4)

# ── MAIN LOOP ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cv2.namedWindow("Pose2Minecraft", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose2Minecraft", 1080, 720)
print("[control] Starting. ESC to quit or move mouse to top-left to abort.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt  = now - last_time
    last_time = now

    img = cv2.flip(frame, 1)
    # Resize video frame for larger display
    img = cv2.resize(img, (1080, 720), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        # 1) extract skeleton
        sk = np.array([
            [
              res.pose_landmarks.landmark[l.value].x,
              res.pose_landmarks.landmark[l.value].y
            ]
            for l in LANDMARKS
        ])

        # 2) build features
        norm = normalize_skeleton(sk)
        flat = norm.flatten()
        geom = extract_geom(norm)
        delta = temporal_delta(prev_norm, norm) if prev_norm is not None else np.zeros_like(flat)
        prev_norm = norm
        feat = np.hstack([flat, geom, delta]).reshape(1, -1)

        # 3) raw prediction
        pred_frame = clf.predict(feat)[0]

        # ── DEBOUNCE ───────────────────────────────
        if pred_frame != last_frame_pred:
            last_frame_pred  = pred_frame
            frame_pred_start = now
        elif (stable_pred != pred_frame and
              now - frame_pred_start >= IGNORE_THRESHOLD):
            # pose has been stable long enough → commit to stable_pred
            stable_pred = pred_frame

        # ── ACTION MAPPING (on stable_pred) ────────
        if stable_pred != prev_stable_pred:
            # exiting previous pose
            if prev_stable_pred == "t_pose" and w_down:
                pyautogui.keyUp("w")
                w_down = False

            # entering new pose
            if stable_pred == "t_pose":
                pyautogui.keyDown("w")
                w_down = True

            # up/down pattern tracking
            if stable_pred in ("disco_up","disco_down"):
                updown_hist.append(stable_pred)
                if list(updown_hist) == ["disco_up","disco_down","disco_up","disco_down"]:
                    pyautogui.mouseDown()
                    click_down = True
                    click_start = now
            else:
                updown_hist.clear()

            prev_stable_pred = stable_pred

        # continuous actions for the current stable pose
        if stable_pred == "disco_left":
            pan_mouse(-MOUSE_SPEED, dt)
        elif stable_pred == "disco_right":
            pan_mouse( MOUSE_SPEED, dt)

        # release click after 1s
        if click_down and (now - click_start) >= CLICK_HOLD_TIME:
            pyautogui.mouseUp()
            click_down = False

        # ── DRAW OVERLAY ──────────────────────────
        h, w = img.shape[:2]
        for lm_enum in LANDMARKS:
            lm = res.pose_landmarks.landmark[lm_enum.value]
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (0,255,255), -1)
        for s,e in CONNECTIONS:
            ls, le = res.pose_landmarks.landmark[s.value], res.pose_landmarks.landmark[e.value]
            xs, ys = int(ls.x*w), int(ls.y*h)
            xe, ye = int(le.x*w), int(le.y*h)
            cv2.line(img, (xs,ys), (xe,ye), (255,255,0), 2)

        cv2.putText(img, stable_pred or "", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    cv2.imshow("Pose2Minecraft", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ── CLEANUP ───────────────────────────────────────────────────────────
if w_down:     pyautogui.keyUp("w")
if click_down: pyautogui.mouseUp()
cap.release()
cv2.destroyAllWindows()
