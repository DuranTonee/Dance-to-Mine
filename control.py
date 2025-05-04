import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyautogui
import mouse
import time
from collections import deque
import threading

# ── CONFIG ─────────────────────────────────────────────────────────────
MODEL_PATH       = "pose_clf.pkl"
IGNORE_THRESHOLD = 0.2    # seconds to ignore very short detections
ROTATE_DURATION  = 0.4
MOUSE_SPEED      = 750    # px per second (was 500)
CLICK_HOLD_TIME  = 1.5    # seconds (was 1.0)

SCROLL_INTERVAL  = 1.0
last_scroll      = 0.0
DROP_INTERVAL    = 1.0
last_drop        = 0.0

sprint_down      = False
crouch_down      = False

mouse_speed      = 0.0
vertical_speed   = 0.0
keep_moving      = True

rotate_start     = 0.0
rotating         = False

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

# ── FEATURE HELPERS ────────────────────────────────────────────────────
## recenters all points on the hip midpoint and scales them by torso length so that people of different sizes map to the same feature space.
def normalize_skeleton(skel):
    hip_l, hip_r = skel[6], skel[7]
    center = (hip_l + hip_r) / 2
    coords = skel - center
    sh_l, sh_r = skel[0], skel[1]
    torso = np.linalg.norm(((sh_l + sh_r)/2) - center) + 1e-6
    return coords / torso

## names says it. distance between 2 points
def distance(a, b):
    return np.linalg.norm(a - b)

## computes the angle at point b formed by the segments b→a and b→c, in degrees.
def angle(a, b, c):
    v1, v2 = a - b, c - b
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

## picks two geometric features (one distance, one angle) for your classifier.
def extract_geom(skel):
    return [
        distance(skel[4], skel[2]),        # wrist–elbow
        angle(skel[0], skel[2], skel[4])   # shoulder–elbow–wrist angle
    ]

## captures motion by subtracting last frame’s normalized coords.
def temporal_delta(prev, curr):
    return (curr - prev).flatten()

# ── CONTROL HELPERS ────────────────────────────────────────────────────
def _mouse_mover():
    """Background thread: small moves at 100Hz based on global mouse_speed."""
    hz = 100.0
    delay = 1.0 / hz
    while keep_moving:
        dx = mouse_speed
        dy = vertical_speed
        if dx or dy:
           # move (dx,dy) per second split over hz ticks
           mouse.move(dx/hz, dy/hz, absolute=False)
        time.sleep(delay)

def toggle_sprint():
    global sprint_down
    if not sprint_down:
        pyautogui.keyDown("shift")
        sprint_down = True
    else:
        pyautogui.keyUp("shift")
        sprint_down = False

def toggle_crouch():
    global crouch_down
    if not crouch_down:
        pyautogui.keyDown("ctrl")
        crouch_down = True
    else:
        pyautogui.keyUp("ctrl")
        crouch_down = False

def do_jump():
    """Hold space for 1 second."""
    pyautogui.keyDown("space")
    time.sleep(1)
    pyautogui.keyUp("space")

def delayed_stop_w():
    """Release W 1s after t_pose ends."""
    global w_down
    time.sleep(1)
    pyautogui.keyUp("w")
    w_down = False

# start the thread once
threading.Thread(target=_mouse_mover, daemon=True).start()

# ── SETUP MODEL & STATE ────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
clf = joblib.load(MODEL_PATH)
pyautogui.FAILSAFE = True   # move mouse to top-left to abort

# Debounce & action state
last_frame_pred   = None            # the very last raw prediction
frame_pred_start  = 0.0             # when it started
stable_pred       = None            # the debounced “stable” pose
prev_stable_pred  = None            # previous stable pose to detect transitions
prev_norm         = None            # previous normalized skeleton for computing deltas
last_time         = time.perf_counter()
w_down            = False
click_down        = False
click_start       = 0.0
updown_hist       = deque(maxlen=4) # a small history of “up/down” gestures for click detection

# ── MAIN LOOP ──────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)   # for me it varies 0/2 depending on the camera; 0 is default
cv2.namedWindow("Pose2Minecraft", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose2Minecraft", 1080, 720)
print("[control] Starting. ESC to quit or move mouse to top-left to abort.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.perf_counter()
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
                # if next pose should keep W held a bit longer, delay release
                if stable_pred in ("jump","disco_left","disco_right","hands_up","hands_hips"):
                    threading.Thread(target=delayed_stop_w, daemon=True).start()
                else:
                    # otherwise release immediately
                    pyautogui.keyUp("w")
                    w_down = False

            # entering new pose
            if stable_pred == "t_pose":
                pyautogui.keyDown("w")
                w_down = True

            # up/down pattern tracking
            if stable_pred in ("disco_up","disco_down"):
                updown_hist.append(stable_pred)
                hist = list(updown_hist)           # ← convert to list
                if not click_down and len(hist) >= 2 and hist[-2:] == ["disco_up","disco_down"]:
                    pyautogui.mouseDown()
                    click_down = True
            else:
                if click_down:
                    pyautogui.mouseUp()
                    click_down = False
                updown_hist.clear()
                    
            # leg_left toggles sprint in a background thread. NOTE: I have sprinting set to "SHIFT", not "CTRL"
            if stable_pred == "leg_left":
                threading.Thread(target=toggle_sprint, daemon=True).start()
            
            # squat toggles CTRL on/off
            if stable_pred == "squat":
                threading.Thread(target=toggle_crouch, daemon=True).start()
            
            # jump holds space for 1 second
            if stable_pred == "jump":
                threading.Thread(target=do_jump, daemon=True).start()
            
            prev_stable_pred = stable_pred

        # leg_right → one scroll-down per second
        if stable_pred == "leg_right":
            if now - last_scroll >= SCROLL_INTERVAL:
                pyautogui.scroll(-100)    # negative is scroll down
                last_scroll = now

        # continuous actions for the current stable pose
        # update the global pan speed
        if stable_pred == "disco_left":
            mouse_speed = -MOUSE_SPEED
            rotating    = True
            rotate_start = now
        elif stable_pred == "disco_right":
            mouse_speed =  MOUSE_SPEED
            rotating    = True
            rotate_start = now
        # look up/down with hands
        elif stable_pred == "hands_up":
            vertical_speed = -MOUSE_SPEED
            rotating       = True
            rotate_start   = now
        elif stable_pred == "hands_hips":
            vertical_speed =  MOUSE_SPEED
            rotating       = True
            rotate_start   = now
        else:
            mouse_speed = 0.0
            vertical_speed = 0.0

        if rotating and (now - rotate_start) >= ROTATE_DURATION:
            mouse_speed    = 0.0
            vertical_speed = 0.0
            rotating       = False



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
        # display sprint status
        status = "SPRINT ON" if sprint_down else "SPRINT OFF"
        cv2.putText(img, status, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)
        # display crouch (CTRL) status
        crouch_status = "CROUCH ON" if crouch_down else "CROUCH OFF"
        cv2.putText(img, crouch_status, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    cv2.imshow("Pose2Minecraft", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ── CLEANUP ───────────────────────────────────────────────────────────
if w_down: pyautogui.keyUp("w")
if click_down: pyautogui.mouseUp()
if crouch_down: pyautogui.keyUp("ctrl")
if sprint_down: pyautogui.keyUp("shift")
keep_moving = False     # signal thread to stop
cap.release()
cv2.destroyAllWindows()