import cv2
import numpy as np
from pathlib import Path
import sys
import whisper
import speech_recognition as sr
import time
from typing import Callable, Optional, Any
import multiprocessing

# My imports
from demo_lib import (
    CSEducationHandler,
    VoiceConversationHandler,
    default_response_handler,
)
from constants import (
    DEFAULT_MODEL,
    DEFAULT_MICROPHONE_LINUX,
    DEFAULT_NON_ENGLISH,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_RECORD_TIMEOUT,
    DEFAULT_PHRASE_TIMEOUT,
    DEFAULT_INITIAL_PHRASE_TIMEOUT,
    DEFAULT_NUM_CONVERSATIONAL_STEPS,
    DEFAULT_SLEEP_AFTER_RESPONSE,
)


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
def cv2_spinner():
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
    while True:
        if play_frames(frames, fperiod_clipped_ms):
            break
        cv2.waitKey(wait_after_video_period_clipped_ms)
    cv2.destroy_all_windows()


def tts_responder(
    # Model parameters
    model: str = DEFAULT_MODEL,
    non_english: bool = DEFAULT_NON_ENGLISH,
    # Speech recognition parameters
    phrase_timeout: float = DEFAULT_PHRASE_TIMEOUT,
    initial_phrase_timeout: float = DEFAULT_INITIAL_PHRASE_TIMEOUT,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    record_timeout: float = DEFAULT_RECORD_TIMEOUT,
    num_conversational_steps: int = DEFAULT_NUM_CONVERSATIONAL_STEPS,
    sleep_after_response: float = DEFAULT_SLEEP_AFTER_RESPONSE,
    # In linux you apparently should specify a microphone? look below (not my code originally)
    default_microphone_linux: str = DEFAULT_MICROPHONE_LINUX,
    # Response handling
    handle_response: Callable[
        [str, VoiceConversationHandler], None
    ] = default_response_handler,
) -> None:
    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if "linux" in sys.platform:
        mic_name = default_microphone_linux
        if not mic_name or mic_name == "list":
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'Microphone with name "{name}" found')
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    if model != "large" and not non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    print("Model loaded.\n")

    # Make a ChatGPT handler to respond to our questions
    handler = CSEducationHandler()
    voiceHandler = VoiceConversationHandler(
        phrase_timeout,
        initial_phrase_timeout,
        energy_threshold,
        record_timeout,
        audio_model,
        source,
    )

    num_conversational_steps = num_conversational_steps
    assert num_conversational_steps == 0 or num_conversational_steps > 0
    at_step = 0
    while (num_conversational_steps == 0) or at_step < num_conversational_steps:
        try:
            print("Listening...")
            contents = voiceHandler.listen()
            if len(contents.strip()) < 1:
                print("Heard nothing.")
            else:
                print("Responding...")
                response = handler.request(contents)
                handle_response(response, voiceHandler)
                at_step += 1
            time.sleep(sleep_after_response)
        except KeyboardInterrupt:
            break


# cv2_spinner()
# tts_responder()
if __name__ == "__main__":
    cv2_process = multiprocessing.Process(target=cv2_spinner, args=())
    tts_process = multiprocessing.Process(target=tts_responder, args=())

    # Set as daemon processes so that they cannot spawn there own AND they are automatically to be terminated if this is terminated
    cv2_process.daemon = True
    tts_process.daemon = True

    cv2_process.start()
    tts_process.start()

    # If you don't run this then the processes will automatically shut down since the parent dies
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating...")
        # This does seem to be redundant but I guess doesn't hurt; was suggested by ChatGPT
        cv2_process.terminate()
        tts_process.terminate()
        cv2_process.join()
        tts_process.join()


# XXX launch the other two subprocesses
