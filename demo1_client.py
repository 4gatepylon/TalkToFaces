# NOTE: this is heavily based on this: https://github.com/davabase/whisper_real_time/blob/master/transcribe_demo.py
# NOTE this demo is focused on CS concept explanation
# TODO(Adriano) (1) get better support for threading and thread safety, not sure how to use asyncio for this, (2) state machine where you CAN interrupt

import argparse
import os
import speech_recognition as sr
import whisper
import time

from sys import platform

# My library
from demo_lib import CSEducationHandler, VoiceConversationHandler

# This is so that we don't get some errors later!
assert "OPENAI_API_KEY" in os.environ
assert platform == "darwin"


def main():
    # 1. Define and parse arguments
    parser = argparse.ArgumentParser()
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
        "--num-conversational-steps",
        default=1,
        help="How many conversational steps to take. If 0, then the conversation is arbitrarily long.",
        type=int,
    )
    if "linux" in platform:
        parser.add_argument(
            "--default_microphone",
            default="pulse",
            help="Default microphone name for SpeechRecognition. "
            "Run this with 'list' to view available Microphones.",
            type=str,
        )
    args = parser.parse_args()

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if "linux" in platform:
        mic_name = args.default_microphone
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
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)
    print("Model loaded.\n")

    # Make a ChatGPT handler to respond to our questions
    handler = CSEducationHandler()
    voiceHandler = VoiceConversationHandler(
        args.phrase_timeout,
        args.initial_phrase_timeout,
        args.energy_threshold,
        args.record_timeout,
        audio_model,
        source,
    )

    num_conversational_steps = args.num_conversational_steps
    assert num_conversational_steps == 0 or num_conversational_steps > 0
    at_step = 0
    while (num_conversational_steps == 0) or at_step < num_conversational_steps:
        try:
            print("Listening...")
            contents = voiceHandler.listen()
            print("Responding...")
            response = handler.request(contents)
            voiceHandler.say(response)
            at_step += 1
            time.sleep(0.1)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
