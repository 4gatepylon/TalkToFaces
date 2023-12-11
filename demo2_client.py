# XXX test

import requests
import cv2
import numpy as np
import pygame
import os
import tempfile
import argparse
import tempfile

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
    pygame.mixer.music.load(mp4_file_path)
    pygame.mixer.music.play()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame color to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)  # Rotate frame

        # Convert frame to Pygame surface and display it
        frame_surface = pygame.surfarray.make_surface(frame)
        window = pygame.display.get_surface()
        window.blit(frame_surface, (0, 0))
        pygame.display.update()

        # Check for quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return

    cap.release()
    pygame.quit()


def main(server_url: str) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_url",
        help="Server URL which should be able to recieve HTTP and do the mp3 => mp4 stuff",
        type=str,
    )

    mp3_file_path = "path/to/your/mp3/file.mp3"  # Replace with your MP3 file path
    server_url = "http://localhost:5000/upload"  # Replace with your Flask server URL

    mp4_file = send_mp3_and_receive_mp4(mp3_file_path, server_url)
    if mp4_file:
        play_mp4(mp4_file)
        os.remove(mp4_file)  # Clean up the temporary file
