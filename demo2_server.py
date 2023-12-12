# XXX test!

from flask import Flask, request, send_file
import shutil
from werkzeug.utils import secure_filename
from pathlib import Path

import sda  # NOTE this is from the speech-driven-animation shit

app = Flask(__name__)

# NOTE save all the mp3 files to ~/mp3/server

# This is a simple server that takes in an mp3 file and then uses the speech-driven-animation repository to
# turn it into an MP4 of an image (that is the hardcoded image for now: the bmp) to create an MP4 video of
# that image (supposed to be a head) saying those words.


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ["mp3"]


# To test do (assuming you've done export REMOTE_IP=... and have audiocapture.mp3 in .)
# "curl -X POST -F "file=@audiocapture.mp3" http://$REMOTE_IP/convert -o result-from-server.mp4"
@app.route("/convert", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    print("Got file")  # Debug

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Saving file to path
        print("Saving file to path")
        mp3_server_path = (
            Path("/home/adriano") / "mp3-server"
        )  # TODO(Adriano) don't just hard-code this plz
        shutil.rmtree(mp3_server_path.as_posix(), ignore_errors=True)
        mp3_server_path.mkdir(parents=True, exist_ok=True)
        mp3_path = mp3_server_path / filename
        file.save(mp3_path.as_posix())
        assert mp3_path.exists()
        print("\n".join((x.as_posix() for x in mp3_server_path.iterdir())))  # Debug
        print("Done!")

        # Now create the mp3
        print("Instantiating Video Animator")
        va = sda.VideoAnimator(gpu=0)  # Instantiate the animator
        print("Running...")
        vid, aud = va(
            "speech-driven-animation-master/sample_face.bmp",
            # "speech-driven-animation-master/sample_audio.wav",
            mp3_path.as_posix(),
        )  # NOTE this is hardcoded
        print("Saving video!")
        genpath = Path(
            "/home/adriano/generated.mp4"
        ).expanduser()  # TODO(Adriano) don't hardcode this!
        try:
            genpath.unlink()
        except FileNotFoundError:
            pass
        va.save_video(vid, aud, genpath.as_posix())
        print(f"Done (in {genpath.as_posix()})")

        # Don't clear anything yet so we can debug
        return send_file(genpath.as_posix(), as_attachment=True)


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
