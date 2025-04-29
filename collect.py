import os
import csv
import cv2
import mediapipe as mp

# ── CONFIG ────────────────────────────────────────────────────────────────
# Which landmarks to record (x,y). Feel free to add/remove joints.
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
CSV_FILE = "data.csv"

# ── SETUP mediapipe ────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
    """Open webcam, detect pose, write selected landmarks + label per frame."""
    cap = cv2.VideoCapture(0)
    ensure_header()

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        print(f"[collect] Recording '{class_name}'. Press 'q' to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip & convert color
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run pose detection
            results = pose.process(rgb)

            if results.pose_landmarks:
                row = [class_name]
                for lm_enum in LANDMARKS:
                    lm = results.pose_landmarks.landmark[lm_enum.value]
                    row += [lm.x, lm.y]

                writer.writerow(row)

                # Overlay live feedback
                cv2.putText(frame, class_name, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Collecting poses", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    label = input("Enter pose label to collect (e.g. 'T_pose'): ").strip()
    collect(label)
