"""
ArtWorks filter website.

Upload an image, pick one of Guilherme's filters, press Run.
The filters in ../ArtWorks/Guilherme are run UNCHANGED, exactly like on the
command line:   python <filter>.py imagename.jpg
Each run happens in its own temporary folder, so the Guilherme/Output folder
is never touched.

Run:   python app.py      then open http://127.0.0.1:5000
"""
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

HERE = Path(__file__).resolve().parent
FILTER_DIR = (HERE.parent / "Guilherme").resolve()

# label shown in dropdown -> (script, suffix the script adds to the output name)
FILTERS = {
    "Duotone Pop Art":  ("popart_duotone.py", "_popart"),
    "Red Pop":          ("Red_Pop.py",        "_redpop"),
    "Vintage":          ("vintage_filter.py", "_vintage"),
}

ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESULTS = {}  # job id -> output png path

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB upload limit


@app.route("/")
def index():
    return render_template("index.html", filters=list(FILTERS))


@app.route("/run", methods=["POST"])
def run():
    file = request.files.get("image")
    name = request.form.get("filter", "")
    if file is None or not file.filename:
        return jsonify(error="Please choose an image first."), 400
    if name not in FILTERS:
        return jsonify(error="Unknown filter."), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported file type {ext or '(none)'}."), 400

    script, suffix = FILTERS[name]
    work = Path(tempfile.mkdtemp(prefix="artworks_"))
    try:
        shutil.copy(FILTER_DIR / script, work / script)
        src = work / f"upload{ext}"
        file.save(src)

        proc = subprocess.run(
            [sys.executable, script, src.name],
            cwd=work, capture_output=True, text=True, timeout=600,
        )
        out = work / "Output" / f"{src.stem}{suffix}.png"
        if proc.returncode != 0 or not out.exists():
            msg = (proc.stderr or proc.stdout or "Filter failed.").strip()
            return jsonify(error=msg), 500

        job = uuid.uuid4().hex
        keep = Path(tempfile.gettempdir()) / f"artworks_{job}.png"
        shutil.move(out, keep)
        RESULTS[job] = keep
        return jsonify(url=f"/result/{job}")
    except subprocess.TimeoutExpired:
        return jsonify(error="The filter took too long (10 min limit)."), 500
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.route("/result/<job>")
def result(job):
    path = RESULTS.get(job)
    if not path or not path.exists():
        return "Not found", 404
    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
