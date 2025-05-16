import os
import csv
import cv2
import mediapipe as mp
import time

# ── CONFIG ────────────────────────────────────────────────────────────────
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
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,      mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE,     mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,     mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE,    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]
CSV_FILE = "static/animations.csv"
FPS = 30
FRAME_INTERVAL = 1.0 / FPS

mp_pose = mp.solutions.pose

def ensure_header():
    """Overwrite CSV_FILE with header row if missing."""
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    header = ["label"]
    for lm in LANDMARKS:
        header += [f"{lm.name.lower()}_x", f"{lm.name.lower()}_y"]
    # Only write header if file doesn't exist or is empty
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def record_animation(label: str):
    ensure_header()
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("[error] Cannot open webcam.")
        return

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        # 3‐second countdown
        print("[record] Starting in:")
        for sec in range(3, 0, -1):
            print(f"  {sec}...")
            time.sleep(1)
        print(f"[record] Recording '{label}'. Press 'q' to stop.")

        next_frame_time = time.time()
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            while True:
                # throttle to target FPS
                now = time.time()
                if now < next_frame_time:
                    time.sleep(next_frame_time - now)
                next_frame_time += FRAME_INTERVAL

                ret, frame = cap.read()
                if not ret:
                    print("[error] Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)  # mirror
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    # write CSV row
                    row = [label]
                    for lm_enum in LANDMARKS:
                        lm = results.pose_landmarks.landmark[lm_enum.value]
                        row += [lm.x, lm.y]
                    writer.writerow(row)

                    # draw joints
                    for lm_enum in LANDMARKS:
                        lm = results.pose_landmarks.landmark[lm_enum.value]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                    # draw skeleton
                    for start, end in CONNECTIONS:
                        lm_s = results.pose_landmarks.landmark[start.value]
                        lm_e = results.pose_landmarks.landmark[end.value]
                        xs, ys = int(lm_s.x * w), int(lm_s.y * h)
                        xe, ye = int(lm_e.x * w), int(lm_e.y * h)
                        cv2.line(frame, (xs, ys), (xe, ye), (255, 0, 0), 2)

                    # overlay label
                    cv2.putText(
                        frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

                cv2.imshow("Animation Maker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_label = input("Enter label name for this recording: ").strip()
    if not user_label:
        print("[error] Label cannot be empty.")
    else:
        record_animation(user_label)
