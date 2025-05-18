# Dance-to-Mine

---

## Description

Dance-to-Mine (very original, huh?) is a Python-based dance-move recognition pipeline that turns your body poses into game actions. Initially focused on controlling Minecraft via MediaPipe landmarks and a RandomForest model, it’s grown into a full suite:

> **Inspired by**: Fundy’s [Coding Minecraft to work with Dance Moves…](https://www.youtube.com/watch?v=z2sGFFXuu38)

---

## Features

- **Pose Collection**: countdown, CSV logging, and simple OpenCV UI (`collect.py`)  
- **Animation Recording**: custom script for capturing dance animations to CSV for advanced visualizations (`animation_maker.py`)  
- **Feature Engineering**:  
  - Translation & scale normalization  
  - Relative distances & angles  
  - Temporal delta features  
- **Model Training**: 5-fold stratified cross-validation + final RandomForest model (`train.py`)  
- **Real-Time CLI Classification**: skeleton overlay, confidence scores, resizable window (`classify.py`)  
- **Web Dashboard & Live Demo**: Flask + D3.js interface for browsing data, live pose classification, and interactive skeletons (`app.py`, `templates/`, `live.html`)  
- **Pose Notes**: random, playful notes per pose served from `pose_notes.json`
  > Note: front-end was made with ChatGPT
- **Game Control (Minecraft & Beyond)**: map poses to keystrokes/mouse actions via PyAutoGUI & `mouse` (`control.py`)  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- `venv` for environment isolation  
- Webcam  

> [!NOTE]
> The default camera index in most scripts is set to `2`. On many systems it’ll be `0` or `1`—adjust the `cv2.VideoCapture(...)` argument as needed.

### Installation

```bash
# Clone the repo
git clone https://github.com/DuranTonee/Dance-to-Mine.git
cd Dance-to-Mine

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> [!IMPORTANT]
> For the web dashboard and live demo, an internet connection is needed to load D3.js and PapaParse from their CDNs.

---

## Usage

### 1. Collect Pose Data

```bash
python collect.py <pose_label>
```

- Enters a 3 s countdown
- Press **q** to stop recording
- Appends normalized landmarks + label to `static/data.csv`

### 2. Record Dance Animations

```bash
python animation_maker.py <animation_label>
```

- Similar to `collect.py` but logs only selected joints + draws full-body skeleton
- Outputs to `static/animations.csv`

### 3. Train the Model

```bash
python train.py
```

- Runs 5-fold CV, reports fold accuracies
- Trains final RandomForest on all data, saves `pose_clf.pkl`

### 4. CLI Pose Classification

```bash
python classify.py
```

- Opens a resizable window (1.2× enlarged) with live webcam feed
- Draws 12 key joints & bones, shows predicted move + confidence

### 5. Game Control

```bash
python control.py
```

- Uses the trained model to map stable poses → keystrokes/mouse events
- Controls Minecraft (or other games/apps) via PyAutoGUI & `mouse`

> [!NOTE]
> You can abort at any time by moving your mouse to the top-left corner (PyAutoGUI failsafe).

### 6. Web Dashboard & Live Demo

```bash
python app.py
```

- Visit `http://localhost:5000` for neon-retro landing  
- **/gallery**: browse collected frames by pose  
- **/animations**: view all recorded animations with neon effects  
- **/live**: interactive live pose classification + random pose notes  

> [!TIP]
> Edit `static/pose_notes.json` to customize notes shown for each pose.

---

## Project Structure

```
Dance-to-Mine/
├── animation_maker.py # Record dance animations → CSV
├── collect.py # Pose collection script
├── train.py # Feature engineering & model training
├── classify.py # Real-time CLI pose classification
├── control.py # Map poses → game controls
├── app.py # Flask dashboard & live demo
├── static/
│ ├── data.csv # Collected pose data
│ ├── animations.csv # Recorded animations
│ ├── pose_clf.pkl # Trained RandomForest model
│ ├── pose_notes.json # Notes per pose for live demo
│ └── ... # CSS, JS, assets
├── templates/ # Flask HTML templates
│ ├── main.html
│ ├── gallery.html
│ ├── animations.html
│ ├── live.html
│ └── ...
├── requirements.txt
├── LICENSE
├── landmarks.png              # Diagram of selected landmarks
└── teachable_machine-fail/    # Experiments with Teachable Machine
```


---

## How It Works

1. **Collect & Animate**  
   - MediaPipe Pose → select 12 joints → normalize & save to CSV  
   - `animation_maker.py` produces time-series data for D3.js animations  

2. **Feature Engineering**  
   - Center at hip midpoint, scale by torso length  
   - Compute joint-to-joint distances & angles  
   - Frame-to-frame deltas for motion context  

3. **Training**  
   - StratifiedKFold CV → evaluate → final RandomForest pipeline  
   - Model saved as `pose_clf.pkl`  

4. **Real-Time Classification**  
   - Live webcam frames → MediaPipe → feature pipeline → RandomForest → pose prediction  

5. **Web & Visualization**  
   - Flask serves data & D3.js powers neon-style skeleton animations  
   - Interactive /live page for human-in-the-loop demos  

6. **Game Control**  
   - Debounce stable poses → map to keyboard/mouse commands  
   - Control Minecraft (forward, jump, rotate, click) or any app  

---

## License

This project is licensed under the GNU General Public License v3.0.  
See the LICENSE file for details.
