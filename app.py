import io
import pandas as pd
import matplotlib.pyplot as plt
from flask import (
    Flask, render_template, request,
    send_file, url_for, redirect
)

app = Flask(__name__)

# ── LOAD & CONFIG ──────────────────────────────────────────────────────
DATA_FILE = "data.csv"
# Load once at startup and keep original frame indices
data = pd.read_csv(f"static/{DATA_FILE}").reset_index()
# 'index' column is now the frame index
POSE_LABELS = data['label'].drop_duplicates().tolist()

# ── ROUTES ──────────────────────────────────────────────────────────────

@app.route('/')
def main():
    """Landing page with neon-retro branding & links."""
    return render_template('main.html')

@app.route('/gallery')
def gallery():
    """Listing of all poses (labels)."""
    return render_template('gallery.html', labels=POSE_LABELS)

@app.route('/pose/<label>')
def pose_view(label):
    """Show first 30 frames + JS hook for loading more."""
    df_label = data[data['label'] == label]
    frames = df_label['index'].tolist()
    first_frames = frames[:30]
    remaining_frames = frames[30:]
    return render_template(
        'pose.html',
        label=label,
        first_frames=first_frames,
        remaining_frames=remaining_frames,
    )

@app.route('/plot.png')
def plot_png():
    """Draw a single skeleton frame as PNG."""
    frame = int(request.args.get('frame', 0))
    row   = data[data['index'] == frame].iloc[0]

    fig, ax = plt.subplots()
    # Draw bones
    CONNECTIONS = [
        ('left_shoulder','left_elbow'), ('left_elbow','left_wrist'),
        ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
        ('left_shoulder','left_hip'),   ('right_shoulder','right_hip'),
        ('left_hip','left_knee'),       ('left_knee','left_ankle'),
        ('right_hip','right_knee'),     ('right_knee','right_ankle'),
    ]
    for a, b in CONNECTIONS:
        xs = [row[f"{a}_x"], row[f"{b}_x"]]
        ys = [row[f"{a}_y"], row[f"{b}_y"]]
        ax.plot(xs, ys, 'o-', linewidth=3, markersize=6)

    ax.invert_yaxis()
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/delete_frames', methods=['POST'])
def delete_frames():
    """Bulk-delete selected frames and persist."""
    frames = list(map(int, request.form.getlist('frames')))
    label  = request.form['label']
    global data
    data = data[~data['index'].isin(frames)].reset_index(drop=True)
    data.to_csv(DATA_FILE, index=False)
    return redirect(url_for('pose_view', label=label))

@app.route('/pipeline')
def pipeline():
    """Your workflow-pipeline page."""
    return render_template('pipeline.html')

@app.route('/skeleton')
def skeleton():
    """Interactive skeleton viewer."""
    data_url = url_for('static', filename='data.csv')
    return render_template('skeleton_viewer.html', data_url=data_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6942, debug=True)
