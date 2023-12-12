# TODO(Adriano) how can we improve the whole situation with the audio being out of sync with the video?

import requests
import cv2
import pygame
import os
import tempfile
import signal
import multiprocessing
import numpy as np
from moviepy.editor import VideoFileClip
from pathlib import Path


def play_frames(
    frames: list[np.ndarray], fperiod: int, exit_keys: str = [ord("q"), 27]
) -> bool:
    """Play a sequence of frames and return whether to exit or not. One key is assumed to be the exit key. Default exit keys are q and esc (27)."""
    for frame in frames:
        cv2.imshow("frame", frame)
        if cv2.waitKey(fperiod) & 0xFF in exit_keys:
            return True
    return False


# Just play an mp4's video portion on loop
# NOTE that this is useful to debug!
def cv2_spinner_proc_main(
    # IPC Communication
    tts_mp3_queue: multiprocessing.Queue,
    playback_completion_event: multiprocessing.Event,
    end_convo_event: multiprocessing.Event,
) -> None:
    # Collect all the frames
    mp4_file_path = Path("~/Downloads/generated.mp4").expanduser()
    cap = cv2.VideoCapture(mp4_file_path.as_posix())
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert num_frames == float(int(num_frames))
    num_frames = int(num_frames)

    frames = [None] * num_frames
    for fidx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames[fidx] = frame

    # Get framerate and choose a little it of time to just stop
    fps = cap.get(cv2.CAP_PROP_FPS)
    fperiod_ms = 1000 / fps
    fperiod_clipped_ms = int(fperiod_ms)
    wait_after_video_period_ms = 1000
    wait_after_video_period_clipped_ms = int(wait_after_video_period_ms)

    cap.release()

    # Keep playing the video on loop
    playing = True

    # You may or may not want to do this?
    # def kill_cv2(signum, frame):
    #     print("Killing cv2")
    #     cv2.destroyAllWindows()

    # signal.signal(signal.SIGTERM, kill_cv2)
    while playing:
        # Block until we have the mp3 filename or everything is over
        # print("--- (video client) --- checking if done with convo") # Debug - can logspam tho
        if end_convo_event.is_set():
            # print("--- (video client) --- done with convo") # Debug - can logspam tho
            playing = False
            break  # Redunant I guess?
        # print("--- (video client) --- waiting for mp3 filename") # Debug - can logspam tho
        try:
            mp3_filename = tts_mp3_queue.get(timeout=0.1)
            assert Path(mp3_filename).exists()

            # XXX do the whole server communication shit
            print("--- (video client) --- playing frames")
            if play_frames(frames, fperiod_clipped_ms):
                break
            cv2.waitKey(wait_after_video_period_clipped_ms)

            # Signal listening to recommence
            print("--- (video client) --- playback compelete")
            playback_completion_event.set()
        except multiprocessing.queues.Empty:
            pass  # Nothing happening here

    cv2.destroyAllWindows()
    print("--- (video client) --- Done destroying cv2 windows!")


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
