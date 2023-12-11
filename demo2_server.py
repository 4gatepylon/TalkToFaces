# NOTE not tested or anything!

from flask import Flask, request, send_file
import subprocess
import os
from werkzeug.utils import secure_filename
from pathlib import Path

import sda # NOTE this is from the speech-driven-animation shit

app = Flask(__name__)

# NOTE save all the mp3 fiels to ~/mp3/server

@app.route('/convert', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        mp3_path = os.path.join('~/mp3-server', filename)
        file.save(mp3_path)

        # Now create the mp3
        print("Instantiating Video Animator")
        va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
        print("Running...")
        vid, aud = va("sample_face.bmp", "sample_audio.wav")
        print("Saving video!")
        Path("generated.mp4").unlink()
        va.save_video(vid, aud, "generated.mp4")
        print("Done (in generated.mp4)")

        return send_file("generated.mp4", as_attachment=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['mp3']

if __name__ == '__main__':
    app.run(debug=True)
