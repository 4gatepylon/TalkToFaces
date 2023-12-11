from __future__ import annotations

import numpy as np
import speech_recognition as sr
import os
import torch
import time
import requests
from pathlib import Path
import logging

import json
from datetime import datetime, timedelta
from queue import Queue
from sys import platform
from typing import Optional, Any


def text2speech(text: str) -> None:
    assert platform == "darwin"

    # https://stackoverflow.com/questions/12758591/python-text-to-speech-in-macintosh
    os.system(f'say "{text}"')


# NOTE this is copied from the Meru Toolbox: https://github.com/Meru-Productions/meru-toolbox/blob/main/QATestingFramework/qa.py#L207
class ChatGPTHandler:
    """Low-level ChatGPT interface. Mainly handles storing history."""

    MODEL = "gpt-3.5-turbo-0613"
    CHAT_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    CONFIG_PATH = Path("gptConfig")

    # Utility
    @staticmethod
    def __create_message(content: str, role: str = "user") -> dict:
        return {"role": role, "content": content}

    @staticmethod
    def __create_system_message(content: str) -> dict:
        return ChatGPTHandler.__create_message(content, "system")

    @staticmethod
    def __create_user_message(content: str) -> dict:
        return ChatGPTHandler.__create_message(content, "user")

    @staticmethod
    def __create_assistant_message(content: str) -> dict:
        return ChatGPTHandler.__create_message(content, "assistant")

    # Might want to add support for config like temperature, multiple messages, and functions, jinja prompts, rag, etc...
    def __init__(
        self,
        model=MODEL,
        keepPreviousMessage: bool = False,
        systemDescription: Optional[str] = None,
        preExistingMessages: Optional[list[dict]] = None,
        verbose: bool = False,
    ):
        # NOTE we only support some of the defaults now
        assert model == ChatGPTHandler.MODEL

        # Load a system description
        self.systemDescription = systemDescription

        # Load pre-existing messages and model
        self.messages = preExistingMessages if preExistingMessages is not None else []
        assert (
            type(self.messages) == list
        )  # Just catch a small error I make when I make None False :/
        self.model = model

        # Set verbose to debug
        self.verbose = verbose

        # Set a flag whether to keep previous messages since it's a common use-case
        self.keepPreviousMessage = keepPreviousMessage

        # Keep some logs that we can use to basically log how much we've spent
        self.charactersSent = 0
        self.requestsSent = 0

    def hasSystemDescription(self) -> bool:
        return self.systemDescription is not None

    def charactersInMessages(self, messages: list[dict]) -> int:
        characters = 0
        for message in messages:
            characters += len(message["content"])
        return characters

    # Main way of interacting with the chat, you will issue a request (currently blocking)
    # and then you will recieve a response and it will be stored for future chats later.
    def request(self, message: str) -> str:
        if self.verbose:
            logging.log(logging.INFO, f"********** Requesting: {message}")

        # Create messages
        requested_message = ChatGPTHandler.__create_user_message(message)
        messages = (
            (
                [ChatGPTHandler.__create_system_message(self.systemDescription)]
                if self.hasSystemDescription()
                else []
            )
            + (self.messages if self.keepPreviousMessage else [])
            + [requested_message]
        )

        # Create the request
        request_headers = {
            "Content-Type": "application/json",
            "Authorization": f'Bearer {os.getenv("OPENAI_API_KEY")}',
            "OpenAI-Organization": os.getenv("OPENAI_ORG"),
        }
        request_body = {
            "messages": messages,
            "model": self.model,
        }

        # Log information for debugging
        if self.verbose:
            logging.log(
                logging.INFO,
                f"Request Headers:\n{json.dumps(request_headers, indent=4)}",
            )
            logging.log(
                logging.INFO, f"Request Body:\n{json.dumps(request_body, indent=4)}"
            )

        # Request and get response
        req = requests.post(
            ChatGPTHandler.CHAT_API_ENDPOINT, headers=request_headers, json=request_body
        )

        response = req.json()
        if self.verbose:
            logging.log(logging.INFO, f"Response:\n{json.dumps(response, indent=4)}")
        response_content = response["choices"][0]["message"]["content"]
        if self.keepPreviousMessage:
            self.messages.append(requested_message)
            self.messages.append(
                ChatGPTHandler.__create_assistant_message(response_content)
            )

        if self.verbose:
            logging.log(logging.INFO, f"********** Responding: {response_content}")

        # Make sure to log this
        _charactersInMessages = self.charactersInMessages(messages)
        self.charactersSent += _charactersInMessages
        self.requestsSent += 1
        logging.log(
            logging.INFO,
            f"Total characters sent: {self.charactersSent} (just sent {_charactersInMessages})",
        )
        logging.log(
            logging.INFO, f"Total requests sent: {self.requestsSent}, (just sent 1)"
        )

        return response_content


class ChatGPTConversationHandler(ChatGPTHandler):
    def __init__(self, conversationSystemDescription: Optional[str] = None):
        # You should really describe what conversation you want
        assert conversationSystemDescription is not None
        super().__init__(
            model=ChatGPTHandler.MODEL,
            keepPreviousMessage=True,
            systemDescription=conversationSystemDescription,
            preExistingMessages=None,
            # we might want to change this later but whatever :P
            verbose=False,
        )


class CSEducationHandler(ChatGPTConversationHandler):
    """An EXAMPLE handler that is meant to have conversations specifically with folks interested
    in learning about CS. This is meant to answer basic questions like "What is SLAM?" or
    "What are some key papers in the field of NLP?" and the like. Short questions, gives reaonably-short
    (as well) answers. It has a prompt that is tayloed to make it a little bit more robust.
    """

    SYS_PROMPT = """You are a helpful AI assistant trying to help, mostly junior software developers, but also the occasional senior software developer, learn Computer Science concepts. Each question from the user is always from the same user and should correspond, most of the time, to some CS concept requiring an explanation. However, you may sometimes just have an open-ended conversation. Try to be friendly and very concise and put things out there but let the user lead the conversation. Concision is important. NOTE: sometimes you will see wierd text because there was a speech to text model used. It is possible that the text given to you is not exactly the correct character-by-character text. If anything is not clear ask the user or infer the most likely option if it is obvious from the full context."""

    def __init__(self):
        super().__init__(CSEducationHandler.SYS_PROMPT)


def record_callback(_, audio: sr.AudioData, data_queue: Queue) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)


class VoiceConversationHandler:
    """
    Right now we have a state machine where the program listens to you and then speaks without listening and then starts listening again.
    """

    def __init__(
        self,
        phrase_timeout: float,
        initial_phrase_timeout: float,
        energy_threshold: float,
        record_timeout: Optional[float],
        audio_model: torch.nn.Module,
        source: Any,
        # While we develop...
        debug: bool = True,
    ) -> None:
        # Timeouts and thresholds, energy must be at least at energy_threshold to record, it will record forever until silence (based on phrase timeout)
        self.initial_phrase_timeout = initial_phrase_timeout
        self.phrase_timeout = phrase_timeout
        self.energy_threshold = energy_threshold

        # Control "how real time" the recording is (unclear exactly what this means, is it a window?)
        self.record_timeout = record_timeout

        # This should be a whisper model or equivalent
        self.audio_model = audio_model

        # Note 100% what this is, got it from their code
        self.source = source

        # This stores the audio so far
        self.data_queue = Queue()

        # This records the voice
        self.recorder = sr.Recognizer()
        with source:
            self.recorder.adjust_for_ambient_noise(source)
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False

        # Look at comment below, NOTE that since this is threaded we can use non-emptiness of queue as a signal for there
        # to be new data to consider; NOTE that the queue is used as a TEMPORARY BUFFER and that we use a queue because
        # there may be more than a single datapoint
        self.record_callback = lambda _, audio: record_callback(
            _, audio, self.data_queue
        )

        # Don't spin too fast :P
        self.processor_loop_sleep = 0.1

        self.debug = debug

    def listen(self) -> str:
        # This is what we'll join in the end
        listened_data: list[str] = []

        # 1. Start Recording in a seperate thread
        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        self.recorder.listen_in_background(
            self.source, self.record_callback, phrase_time_limit=self.record_timeout
        )

        # 2. In this thread, start transcribing and stop once we are finished!
        listen_start_time = datetime.utcnow()
        most_recent_listen_active_time = None
        done_listening: bool = False
        while not done_listening:
            now = datetime.utcnow()

            # 1. Check if we are done listening (pause)
            if (
                most_recent_listen_active_time is None
                and now - listen_start_time
                > timedelta(seconds=self.initial_phrase_timeout)
            ):
                if self.debug:
                    print("INITIAL PHRASE TIMEOUT")
                done_listening = True
            elif (
                most_recent_listen_active_time is not None
                and now - most_recent_listen_active_time
                > timedelta(seconds=self.phrase_timeout)
            ):
                if self.debug:
                    print("PHRASE TIMEOUT")  # XXX
                done_listening = True
            # 2. Process Queue Items
            if self.data_queue.empty():
                pass  # Nothing to do (time is passing by, we'll timeout if we go for too long like this)
            else:
                most_recent_listen_active_time = now

                # Combine audio data from queue and then convert in-ram buffer to something the model can directly use
                # (16bit to 32bit if necessary, clamp audio freq to PCM wavelen. compat. default of 2^15hz max)
                audio_data = b"".join(self.data_queue.queue)
                self.data_queue.queue.clear()
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                # Read the transcription.
                result = self.audio_model.transcribe(
                    audio_np, fp16=torch.cuda.is_available()
                )
                text = result["text"].strip()
                listened_data.append(text)

                # Infinite loops are bad for processors, must sleep.
                time.sleep(self.processor_loop_sleep)

            # 3. Exit if necessary
            if done_listening:
                break

        listened_data_merged = " ".join(listened_data)
        if self.debug:
            print(f'LISTENED "{listened_data_merged}"')
        return listened_data_merged

    def say(self, text: str) -> None:
        text2speech(text)
