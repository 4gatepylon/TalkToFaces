from flask import Flask, request, send_file
import shutil
from werkzeug.utils import secure_filename
from pathlib import Path
import tempfile

import sda  # NOTE this is from the speech-driven-animation shit

app = Flask(__name__)

# This is a simple server that takes in an mp3 file + image and then turns them into an MP4 using our system


# To test do (assuming you've done export REMOTE_IP=... and have audiocapture.mp3 in .)
# "curl -X POST -F "image=@speech-driven-animation-master/sample_face.bmp" -F "audio=@audiocapture.mp3" http://$REMOTE_IP/convert -o result-from-server.mp4"
@app.route("/convert", methods=["POST"])
def upload_file():
    if "image" not in request.files or "audio" not in request.files:
        return "No file part", 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return "No selected file", 400
    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return "No selected file", 400
    if not image_file.filename.endswith(".bmp"):
        return "Image file must be a bmp", 400
    if not audio_file.filename.endswith(".mp3"):
        return "Audio file must be an mp3", 400
    print("Got files")  # Debug

    # We will create an MP4 in a specific file and send it back
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        mp3_filepath = tempdir / "audio.mp3"
        bmp_filepath = tempdir / "image.bmp"
        mp4_filepath = tempdir / "result.mp4"

        print("Saving file to path")  # Debug
        image_file.save(bmp_filepath.as_posix())
        assert bmp_filepath.exists()
        audio_file.save(mp3_filepath.as_posix())
        assert mp3_filepath.exists()

        ### Debug ###
        print("Tempdir:")
        print("\n".join((x.as_posix() for x in tempdir.iterdir())))  # Debug
        print("Done!")
        ### ...

        # Now create the mp3
        print("Instantiating Video Animator")
        va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
        print("Running...")
        # Seems to work OK with MP3's
        vid, aud = va(
            bmp_filepath.as_posix(),
            mp3_filepath.as_posix(),
        )  # NOTE this is hardcoded
        print("Saving video!")
        va.save_video(vid, aud, mp4_filepath.as_posix())
        assert mp4_filepath.exists() and mp4_filepath.is_file()
        print(f"Done (in {mp4_filepath.as_posix()})")

        # Don't clear anything yet so we can debug
        return send_file(mp4_filepath.as_posix(), as_attachment=True)


# To test just do regular old curl to the /ping endpoint on port 80
@app.route("/ping", methods=["GET"])
def ping():
    # Use this route to debug
    print("Ok")
    return "Ok"


# NOTE this server is ... around 20sec latency for serving an mp4 file, so not great but maybe good enough for a really shitty demo :P
if __name__ == "__main__":
    # NOTE: you'll need sudo: I suggest you follow this tutorial and run in virtualenv: https://stackoverflow.com/questions/77144665/how-can-i-run-python-as-root-or-sudo-while-still-using-my-local-pip
    # Listen on all net ifaces: https://stackoverflow.com/questions/30554702/cant-connect-to-flask-web-service-connection-refused
    app.run(host="0.0.0.0", debug=False, port=80)
