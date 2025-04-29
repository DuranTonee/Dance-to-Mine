import io
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, url_for, redirect
import os

app = Flask(__name__)

# ── LOAD & CONFIG ──────────────────────────────────────────────────────
DATA_FILE = "data.csv"
# Load once at startup
data = pd.read_csv(DATA_FILE)
# Unique pose labels
POSE_LABELS = sorted(data['label'].unique())

# Landmarks & connections (match your collect.py)
LANDMARKS = [
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]
CONNECTIONS = [
    ('left_shoulder','left_elbow'), ('left_elbow','left_wrist'),
    ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
    ('left_shoulder','left_hip'),   ('right_shoulder','right_hip'),
    ('left_hip','left_knee'),       ('left_knee','left_ankle'),
    ('right_hip','right_knee'),     ('right_knee','right_ankle'),
]

@app.route('/')
def index():
    # List all pose labels
    return render_template('index.html', labels=POSE_LABELS)

@app.route('/pipeline')
def pipeline():
    return render_template('pipeline.html')

@app.route('/pose/<label>')
def pose_view(label):
    # Filter rows for this label and preserve original frame index
    df = data.reset_index()
    df_label = df[df['label'] == label]
    # List of frame indices
    frames = df_label['index'].tolist()
    return render_template('pose.html', label=label, frames=frames)

@app.route('/plot.png')
def plot_png():
    # Render skeleton for frame index N
    frame = int(request.args.get('frame', 0))
    row   = data.iloc[frame]

    fig, ax = plt.subplots()
    # Draw connections
    for a, b in CONNECTIONS:
        x_vals = [row[f"{a}_x"], row[f"{b}_x"]]
        y_vals = [row[f"{a}_y"], row[f"{b}_y"]]
        ax.plot(x_vals, y_vals, 'o-', linewidth=3, markersize=6)

    ax.invert_yaxis()  # match image coords
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

# Route to delete one frame by index
@app.route('/delete_frame', methods=['POST'])
def delete_frame():
    frame = int(request.form['frame'])
    label = request.form['label']
    global data
    # remove the row and rewrite CSV
    data = data.drop(data.index[frame]).reset_index(drop=True)
    data.to_csv(DATA_FILE, index=False)
    return redirect(url_for('pose_view', label=label))

if __name__ == '__main__':
    app.run(debug=True)