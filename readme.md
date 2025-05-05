# Dance-to-Mine

**TODOs**  
- [ ] Define and implement pose → Minecraft action mappings  
- [ ] Expand and augment dance-move dataset  
- [ ] Add configuration file for custom poses and actions  
- [ ] Improve UI/UX in the Flask dashboard   

---

## Description

Dance-to-Mine (very original, huh?) is a Python-based dance-move recognition pipeline that turns your body poses into game actions - focused on controlling (initially) Minecraft via MediaPipe landmarks and a RandomForest model. Inspired by Fundy's [Coding Minecraft to work with Dance Moves...](https://www.youtube.com/watch?v=z2sGFFXuu38)

---

## Features

- **Pose collection** with countdown, CSV logging, and simple OpenCV UI (`collect.py`)  
- **Feature engineering**:  
  - Translation & scale normalization  
  - Relative distances & angles  
  - Temporal delta features  
- **Model training** with 5-fold stratified cross-validation and a `RandomForestClassifier` (`train.py`)  
- **Real-time classification** with skeleton overlay and red-font labels (`classify.py`)  
- **Flask web dashboard** for browsing collected frames, editing/deleting data, and visualizing poses (`app.py`, `templates/`)  

---

## Getting Started

### Prerequisites

- Python 3.8+  
- `venv` for environment isolation  
- Webcam (for live capture)  

### Installation

```bash
# Clone the repo
git clone https://github.com/DuranTonee/Dance-to-Mine.git
cd Dance-to-Mine

# Create & activate virtual environment
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Collect Data
-------------
```bash
python collect.py
```

- Enter the pose label (e.g., jump, left_turn)
- Perform the move during the 3-second countdown until you press 'q'


Train Model
------------
```bash
python train.py
```

- Runs 5-fold stratified CV, reports fold accuracies
- Trains final model on full data, saves pose_clf.pkl


Classify Live
--------------
```bash
python classify.py
```
- Opens a resizable window with your webcam feed
- Draws only the 12 key joints & bones (no face/fingers)
- Displays the detected move and confidence in red


Dashboard (Optional)
---------------------
```bash
$ python app.py
```

- Browse /pipeline to see the workflow steps
- Click /pose/{label} to review and delete frames
- Visualize skeletons via generated PNGs

## Project Structure
```
Dance-to-Mine/
├── collect.py                 # Data collection script
├── train.py                   # Feature engineering & model training
├── classify.py                # Real-time pose classification
├── app.py                     # Flask dashboard & visualization
├── data.csv                   # Collected landmark data
├── pose_clf.pkl               # Serialized RandomForest pipeline
├── static/                    # CSS, images, frontend assets
├── templates/                 # Flask HTML templates (index, pipeline, pose)
├── landmarks.png              # Diagram of selected landmarks
└── teachable_machine-fail/    # Experiments with Teachable Machine
```

## How it works
Collect
--------
- MediaPipe Pose detects 33 landmarks; we record only 12 (arms, torso, legs)
- Normalized coordinates (0–1) saved alongside the pose label

Train
------
- Normalization: Center skeleton at hip midpoint; scale by torso length
- Geometric features: Compute joint-to-joint distances & key angles
- Temporal deltas: Frame-to-frame landmark changes for motion context
- StratifiedKFold ensures balanced evaluation across pose classes

Classify
---------
- Live webcam frames → MediaPipe → feature pipeline → RandomForest → predicted pose
- Overlay a simplified skeleton and label in an enlarged window

Game Control (Future)
----------------------
- Map each pose label to a Minecraft action (e.g., forward, jump, place_block)
- Send keystrokes or API commands to your Minecraft client

## License

This project is licensed under the GNU General Public License v3.0.
See the LICENSE file for details.