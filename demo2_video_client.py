# TODO(Adriano) how can we improve the whole situation with the audio being out of sync with the video?

import requests
import cv2
import whisper
import pygame
import os
import tempfile
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import io
from pathlib import Path

from sys import platform

# My library
from demo_lib import CSEducationHandler, VoiceConversationHandler

# This is so that we don't get some errors later!
assert "OPENAI_API_KEY" in os.environ
assert platform == "darwin"


def send_mp3_and_receive_mp4(mp3_file_path: str, url: str) -> str:
    # Send MP3 to server and receive MP4
    files = {"file": open(mp3_file_path, "rb")}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        # Save the MP4 file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    else:
        print("Error in file upload:", response.status_code)
        return None


def play_mp4(mp4_file_path: str) -> None:
    # Initialize Pygame for audio
    pygame.init()
    pygame.display.set_mode((1, 1))  # Minimal window

    # Load video file
    cap = cv2.VideoCapture(mp4_file_path)

    # Get audio from the same video file
    # Covert to mp3 file in a bytes io first
    # TODO(just return the mp3)
    video = VideoFileClip(mp4_file_path)
    audio = video.audio
    tmp_mp3 = Path("tmp.mp3")
    try:
        tmp_mp3.unlink()
    except FileNotFoundError:
        pass

    video.audio.write_audiofile(tmp_mp3.as_posix())
    pygame.mixer.music.load(tmp_mp3)
    pygame.mixer.music.play()
    print("Playing!")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)
    fps_down_err = 5
    print("using fps", fps + fps_down_err)
    delay_real = 1000 / (fps + fps_down_err)
    delay_approx = int(delay_real)
    # delay_accrued_error = 0
    assert delay_approx <= delay_real
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame color to RGB (OpenCV uses BGR by default)
        # frame = np.rot90(frame)  # Rotate frame

        # # Convert frame to Pygame surface and display it
        # frame_surface = pygame.surfarray.make_surface(frame)
        # window = pygame.display.get_surface()
        # window.blit(frame_surface, (0, 0))
        # pygame.display.update()

        # # Check for quit events
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         cap.release()
        #         pygame.quit()
        #         return
        cv2.imshow("frame", frame)
        this_delay = delay_approx
        # if delay_accrued_error > 1:
        #     additional_delay = int(delay_accrued_error)
        #     this_delay += additional_delay
        #     delay_accrued_error -= additional_delay
        #     assert delay_accrued_error < 1
        # delay_accrued_error += delay_real - delay_approx
        if cv2.waitKey(this_delay) & 0xFF == ord("q"):
            break

    cap.release()
    pygame.quit()


# This should be a server you can get the mp4's from at port 80
assert "REMOTE_IP" in os.environ


def main() -> None:
    # TODO(Adriano) don't use defaults, support argparsing, also support the whole platform shit from before
    model = "tiny.en"
    audio_model = whisper.load_model(model)
    energy_threshold = 1000
    record_timeout = 2
    phrase_timeout = 3
    initial_phrase_timeout = 9
    source = sr.Microphone(sample_rate=16000)
    default_image_path = "speech-driven-animation-master/sample_face.jpg"
    default_image = cv2.imread(default_image_path)

    server_url = os.environ["REMOTE_IP"]

    handler = CSEducationHandler()
    voiceHandler = VoiceConversationHandler(
        phrase_timeout,
        initial_phrase_timeout,
        energy_threshold,
        record_timeout,
        audio_model,
        source,
    )

    frame = default_image
    fps = 60
    frame_period_real_ms = 1000 / fps
    frame_period_clipped = int(frame_period_real_ms)
    while True:
        cv2.imshow("frame", frame)
        period = frame_period_clipped

        if cv2.waitKey(period) & 0xFF == ord("q"):
            break

    cv2.destroy_all_windows


if __name__ == "__main__":
    main()

# Use this to debug!
# if __name__ == "__main__":
#     play_mp4("result-from-server.mp4")
