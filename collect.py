import os
import csv
import cv2
import mediapipe as mp
import time

# ── CONFIG ────────────────────────────────────────────────────────────────
# Which landmarks to record (x,y). Feel free to add/remove joints.
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
# Draw only these bone connections between the selected joints
CONNECTIONS = [
    # Arms
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_ELBOW),
    (mp.solutions.pose.PoseLandmark.LEFT_ELBOW,    mp.solutions.pose.PoseLandmark.LEFT_WRIST),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_ELBOW),
    (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,    mp.solutions.pose.PoseLandmark.RIGHT_WRIST),
    # Torso
    (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER, mp.solutions.pose.PoseLandmark.LEFT_HIP),
    (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER, mp.solutions.pose.PoseLandmark.RIGHT_HIP),
    # Legs
    (mp.solutions.pose.PoseLandmark.LEFT_HIP,      mp.solutions.pose.PoseLandmark.LEFT_KNEE),
    (mp.solutions.pose.PoseLandmark.LEFT_KNEE,     mp.solutions.pose.PoseLandmark.LEFT_ANKLE),
    (mp.solutions.pose.PoseLandmark.RIGHT_HIP,     mp.solutions.pose.PoseLandmark.RIGHT_KNEE),
    (mp.solutions.pose.PoseLandmark.RIGHT_KNEE,    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE),
]
CSV_FILE = "data.csv"

# ── UTILITIES ─────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose


def ensure_header():
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(CSV_FILE):
        header = ["label"]
        for lm in LANDMARKS:
            header += [f"{lm.name.lower()}_x", f"{lm.name.lower()}_y"]
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def collect(class_name: str):
    """
    1. Prompt countdown so user can prepare
    2. Opens webcam
    3. Runs MediaPipe Pose on each frame
    4. Extracts selected joint (x,y) pairs
    5. Draws only those joints & connections (whole-body skeleton)
    6. Appends label + coords to CSV
    """
    cap = cv2.VideoCapture(0)
    ensure_header()

    # Initialize Pose estimator once inside collect so logs appear after prompt
    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        # Give user time to prepare
        print("[collect] Starting in:")
        for sec in range(3, 0, -1):
            print(f"  {sec}...")
            time.sleep(1)
        print(f"[collect] Recording '{class_name}'. Press 'q' to stop.")

        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror image for natural interaction
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run pose detection
                results = pose.process(rgb)

                if results.pose_landmarks:
                    # Build CSV row
                    row = [class_name]
                    for lm_enum in LANDMARKS:
                        lm = results.pose_landmarks.landmark[lm_enum.value]
                        row += [lm.x, lm.y]
                    writer.writerow(row)

                    # Draw selected joints as circles
                    for lm_enum in LANDMARKS:
                        lm = results.pose_landmarks.landmark[lm_enum.value]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

                    # Draw whole-body skeleton connections
                    for (start, end) in CONNECTIONS:
                        lm_s = results.pose_landmarks.landmark[start.value]
                        lm_e = results.pose_landmarks.landmark[end.value]
                        xs, ys = int(lm_s.x * w), int(lm_s.y * h)
                        xe, ye = int(lm_e.x * w), int(lm_e.y * h)
                        cv2.line(frame, (xs, ys), (xe, ye), (255, 0, 0), 2)

                    # Overlay the class name
                    cv2.putText(
                        frame,
                        class_name,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                # Display the processed frame
                cv2.imshow("Collecting poses", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Prompt user for label before initializing Mediapipe logs
    label = input("Enter pose label to collect (e.g. 'T_pose'): ").strip()
    collect(label)
