import cv2
import numpy as np
from pathlib import Path
import sys
import os
import whisper
import speech_recognition as sr
import tempfile
import time
import argparse
from typing import Callable, Optional, Any
import multiprocessing

# My imports
from demo_listen_lib import (
    CSEducationHandler,
    VoiceConversationHandler,
    # default_response_handler,
    save_mp3_response_handler,
)

# cv2_spinner is a test, it is meant to basically play videos once it gets mp3s without talking to the server or really doing anything (state management testing)
from demo2_video_client import (
    # cv2_spinner_proc_main,
    frame_player_proc_main,
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


def tts_responder_proc_main(
    # IPC communication
    tts_mp3_queue: multiprocessing.Queue,
    tmp_dir: str,
    playback_completion_event: multiprocessing.Event,
    end_convo_event: multiprocessing.Event,
    # File Ops
    save_zip: bool,
    save_to_zip_path: str,
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
    ] = save_mp3_response_handler,
) -> None:
    tmp_dir = Path(tmp_dir)
    assert tmp_dir.exists()
    assert tmp_dir.is_dir()

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

    # Where to store the files when we finish conversing so we can save them happily ever after :)
    save_to_zip_path = Path(save_to_zip_path).expanduser().resolve()
    name = save_to_zip_path.name
    assert name.endswith(".zip")
    save_to_zip_name = name[:-4]
    save_to_parent_dir = save_to_zip_path.parent

    # Make a ChatGPT handler to respond to our questions
    handler = CSEducationHandler()
    voiceHandler = VoiceConversationHandler(
        # Listening parameters
        phrase_timeout,
        initial_phrase_timeout,
        energy_threshold,
        record_timeout,
        audio_model,
        source,
        # ...
        debug=True,  # Prints more
        # File operations
        default_mp3_location=None,  # Never use default mp3 functionality, always save into file handler's location
        use_file_handler=True,
        file_handler_tmp_dir=tmp_dir,
        max_filehandler_files=(
            num_conversational_steps if num_conversational_steps > 0 else 128
        ),
        save_on_termination=save_zip,
        save_to_parent_dir=save_to_parent_dir,
        save_to_zip_name=save_to_zip_name,
    )

    # Initialize the additional information that depends on what type of response handler we are using
    # to enable those response handlers (TODO(Adriano) in a real service there must be a cleaner way!)
    additional_response_handler_args = [tts_mp3_queue]
    additional_response_handler_kwargs = {}

    # Loop
    num_conversational_steps = num_conversational_steps
    assert num_conversational_steps == 0 or num_conversational_steps > 0
    at_step = 0
    while (num_conversational_steps == 0) or at_step < num_conversational_steps:
        # Block until we have completed any previous playback
        print("Waiting for playback to complete...")
        playback_completion_event.wait()
        playback_completion_event.clear()  # Clear means that the next time we wait we will block
        print("Playback is complete!")

        listened_to_something = False
        _exit = False
        while not listened_to_something:
            # Start listening...
            try:
                print("Listening...")
                contents = voiceHandler.listen()
                if len(contents.strip()) < 1:
                    print("Heard nothing.")
                    pass  # dont' change listened to something
                else:
                    listened_to_something = True
                    print("Handling response...")
                    response = handler.request(contents)
                    # NOTE that when you put into the queue, you WILL have the side effect of putting into the queue and now the other process should do the whole playback process!
                    handle_response(
                        response,
                        voiceHandler,
                        *additional_response_handler_args,
                        **additional_response_handler_kwargs,
                    )
                    at_step += 1
                    time.sleep(sleep_after_response)
                    print("Done handling response... will resume")
            except KeyboardInterrupt:
                _exit = True
                break  # Break inner loop (listen until some query comes in loop)
        if _exit:
            break  # Break conversation loop
    time.sleep(0.5)  # TODO(Adriano) don't hardcode this just like that
    print("Done with conversation, waiting for any playback...")
    playback_completion_event.wait()
    end_convo_event.set()
    print("Signalling over for good, really done now...")


if __name__ == "__main__":
    assert "REMOTE_IP" in os.environ
    assert "OPENAI_API_KEY" in os.environ
    assert "FACE" in os.environ

    # 1. Define and parse arguments
    parser = argparse.ArgumentParser()
    # Arguments for how to handle the conversation program
    parser.add_argument(
        # TODO(Adriano) have some form of model streaming
        "--model",
        default="medium",
        help="Model to use",
        choices=["tiny", "base", "small", "medium", "large"],
    )
    parser.add_argument(
        "--non_english", action="store_true", help="Don't use the english model."
    )
    parser.add_argument(
        "--energy_threshold",
        default=1000,
        help="Energy level for mic to detect.",
        type=int,
    )
    parser.add_argument(
        "--record_timeout",
        default=2,
        help="How real time the recording is in seconds.",
        type=float,
    )
    parser.add_argument(
        "--phrase_timeout",
        default=3,
        help="How much empty space between recordings before we "
        "consider it a new line in the transcription.",
        type=float,
    )
    parser.add_argument(
        "--initial_phrase_timeout",
        default=9,  # longer because sometimes you react slowly to the prompt and then the buffer doesn't get filled enough to make a word
        help="How much empty space before giving up on the first reading...",
        type=float,
    )
    parser.add_argument(
        "--num_conversational_steps",
        default=1,
        help="How many conversational steps to take. If 0, then the conversation is arbitrarily long.",
        type=int,
    )
    if "linux" in sys.platform:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
    # Parameters for how to handle SIGTERM
    parser.add_argument(
        "--save_mp3s",
        default=True,
        help="Whether to save mp3s or not into a zipfile upon termination.",
    )
    parser.add_argument(
        "--save_mp3s_to",
        default="~/Downloads/tts-mp3s.zip",
        help="Where to save the mp3s to. Should be in an existing folder and end in .zip",
    )
    args = parser.parse_args()

    assert (
        sys.platform == "darwin"
    )  # No linux since we don't want to deal with this mic shit

    # Make sure to cleanup everything right before we exit
    # NOTE that the zipfile save will occur in the recording_process terminate call and therefore will happen BEFORE we cleanup the tempdir as it should
    with tempfile.TemporaryDirectory() as tempdir:
        mp3_tempdir = Path(tempdir) / "mp3"
        mp4_tempdir = Path(tempdir) / "mp4"  # returned by server
        mp3_tempdir.mkdir()
        mp4_tempdir.mkdir()

        # Queue that is always read
        tts_mp3_queue = multiprocessing.Queue()
        playback_completion_event = multiprocessing.Event()
        end_convo_event = multiprocessing.Event()

        # TODO(Adriano) standardize names next time around
        # Recording process will always be running and alternate between recording and sleeping (during which the playback process will be running)
        # Playback process will always be running and will just show the image until the image is speaking, then it will fetch the video from the server and play it seamlessly
        # into the same window. When it is done, it signals the recording to proess to record anew
        playback_process = multiprocessing.Process(
            # Swap with comment for debug
            # target=cv2_spinner_proc_main,
            # args=(
            #     tts_mp3_queue,
            #     playback_completion_event,
            #     end_convo_event,
            # ),
            target=frame_player_proc_main,
            args=(
                tts_mp3_queue,
                mp4_tempdir,
                playback_completion_event,
                end_convo_event,
            ),
        )
        recording_process = multiprocessing.Process(
            target=tts_responder_proc_main,
            args=(
                # Communicate via queue, signal with events, store mp3s in tmpdir for them
                tts_mp3_queue,
                mp3_tempdir,
                playback_completion_event,
                end_convo_event,
                # Whether or not to save after exit into zipfile
                args.save_mp3s,
                args.save_mp3s_to,
            ),
            kwargs={
                # Model parameters
                "model": args.model,
                "non_english": args.non_english,
                # Speech recognition parameters
                "phrase_timeout": args.phrase_timeout,
                "initial_phrase_timeout": args.initial_phrase_timeout,
                "energy_threshold": args.energy_threshold,
                "record_timeout": args.record_timeout,
                "num_conversational_steps": args.num_conversational_steps,
                # eh...
                "sleep_after_response": DEFAULT_SLEEP_AFTER_RESPONSE,
                # In linux you apparently should specify a microphone? look below (not my code originally)
            },
        )

        # Set as daemon processes so that they cannot spawn there own AND they are automatically to be terminated if this is terminated
        playback_process.daemon = True
        recording_process.daemon = True

        playback_process.start()
        recording_process.start()

        # Signal to start listening
        playback_completion_event.set()

        # If you don't run this then the processes will automatically shut down since the parent dies
        try:
            # NOTE that the conversation will terminate unless we kill it with keyoard
            end_convo_event.clear()
            end_convo_event.wait()
        except KeyboardInterrupt:
            pass
        print("Terminating...")
        # This does seem to be redundant but I guess doesn't hurt; was suggested by ChatGPT
        playback_process.terminate()
        recording_process.terminate()
        playback_process.join()
        recording_process.join()
