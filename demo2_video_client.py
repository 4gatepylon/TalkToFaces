import cv2
import pygame
import os
import tempfile
from PIL import Image
from pillow_heif import register_heif_opener

# https://stackoverflow.com/questions/54395735/how-to-work-with-heic-image-file-types-in-python
register_heif_opener()

import aiohttp
import asyncio
import time
import multiprocessing
import signal
import numpy as np
from moviepy.editor import VideoFileClip
from pathlib import Path
from typing import Optional

# File handler is used by both!
from demo_listen_lib import FileHandler


# Same idea as "curl -X POST -F "image=@speech-driven-animation-master/sample_face.bmp" -F "audio=@audiocapture.mp3" http://$REMOTE_IP/convert -o result-from-server.mp4"
# But async and returns bytes directly for you to do your thing with
async def recv_mp4(bmp_filepath: Path, mp3_filepath: Path) -> bytes:
    assert "REMOTE_IP" in os.environ
    ifp, afp = None, None
    try:
        ifp = open(bmp_filepath.as_posix(), "rb")
        afp = open(mp3_filepath.as_posix(), "rb")
        url = f"http://{os.environ['REMOTE_IP']}/convert"
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field("image", ifp)
            data.add_field("audio", afp)

            async with session.post(url, data=data) as response:
                if response.status == 200:
                    # Read the content of the response (MP4 file)
                    return await response.read()
                else:
                    return None
    finally:
        # Technically not 100% airtight but good enough imo here
        if afp is not None:
            afp.close()
        if ifp is not None:
            ifp.close()


# Should be used once you've recieved the file
def write_mp4_to_tempfile(mp4_bytes: bytes, handler: FileHandler) -> Path:
    mp4_filepath = handler.claim_new_file()
    with open(mp4_filepath.as_posix(), "wb") as f:
        f.write(mp4_bytes)
    return mp4_filepath


def get_frames_from_mp4(mp4_filepath: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(mp4_filepath.as_posix())
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert num_frames == float(int(num_frames))
    num_frames = int(num_frames)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = [None] * num_frames
    for fidx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames[fidx] = frame
    return frames, fps


def play_audio_from_mp4(mp4_filepath: Path, mp3_tempdir: Path) -> None:
    assert mp3_tempdir.exists() and mp3_tempdir.is_dir()
    video = VideoFileClip(mp4_filepath.as_posix())
    audio = video.audio
    mp3_path = mp3_tempdir / "audio.mp3"
    try:
        mp3_path.unlink()
    except FileNotFoundError:
        pass

    video.audio.write_audiofile(mp3_path.as_posix())
    pygame.mixer.music.load(mp3_path)
    pygame.mixer.music.play()


def play_frames(
    frames: list[np.ndarray], fperiod: int, exit_keys: str = [ord("q"), 27]
) -> bool:
    """Play a sequence of frames and return whether to exit or not. One key is assumed to be the exit key. Default exit keys are q and esc (27)."""
    for frame in frames:
        cv2.imshow("frame", frame)
        if cv2.waitKey(fperiod) & 0xFF in exit_keys:
            return True
    return False


# Just play an mp4's video portion on loop; this is for Debug!!
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


### Helper for the stuff below ###
async def await_tts_mp3_filename(
    tts_mp3_queue: multiprocessing.Queue,
    end_convo_event: multiprocessing.Event,
    bmp_filename: Path,
    # CV2 display
    default_image: np.ndarray,
    window_name: str,
) -> tuple[Optional[Path], bool]:
    """Return pathto mp3 (or none) and whether exit signal was sent."""

    # Just loop until we recieve an mp3 filename from the listener tts process
    while True:
        if end_convo_event.is_set():
            return None, True
        try:
            # If a new MP3 is available, we are now going to try to play it
            mp3_filename = tts_mp3_queue.get(timeout=0.1)
            assert Path(mp3_filename).exists()
            return Path(mp3_filename), False

        except multiprocessing.queues.Empty:
            # Just  display the default image
            # Wait 1 ms i.e. very very little, but just enough to DISPLAY
            cv2.imshow(window_name, default_image)
            cv2.waitKey(1)

    # Should never make it here
    raise RuntimeError("Unreachable code")


async def await_mp4_response(
    end_convo_event: multiprocessing.Event,
    bmp_filename: Path,
    mp3_filename: Path,
    mp4_file_handler: FileHandler,
    # CV2 display
    default_image: np.ndarray,
    window_name: str,
) -> tuple[Optional[Path], bool]:
    # Launch a network request
    print(" --- (video client) --- commencing mp4 awaital")
    awaiting_mp4_promise = asyncio.ensure_future(
        recv_mp4(
            Path(bmp_filename),
            Path(mp3_filename),
        )
    )

    # Wait forr the network request and display the relevant (default) image
    print("--- (video client) --- awaital bombs away!")
    while not awaiting_mp4_promise.done():
        if end_convo_event.is_set():
            return None, True
        # Make sure to wait a little to give async http a chance to do its thing
        await asyncio.sleep(0.1)
        # Just as before make sure we display, might be blocking so keep it minimal
        cv2.imshow(window_name, default_image)
        cv2.waitKey(1)

    # Write to a tempfile so the next step can use this
    print("--- (video client) --- awaital done!")
    mp4_filename = write_mp4_to_tempfile(
        awaiting_mp4_promise.result(), mp4_file_handler
    )
    return mp4_filename, False


async def play_mp4_response(
    end_convo_event: multiprocessing.Event,
    mp4_filepath: Path,
    playback_completion_event: multiprocessing.Event,
    mp3_recv_tempdir: Path,
    # CV2 display
    window_name: str,
    default_image: np.ndarray,
    # Give it a little time to finish audio
    wait_after_period: float,
) -> bool:
    playing_frames, fps = get_frames_from_mp4(mp4_filepath)
    fperiod = 1000 / fps
    fperiod_clipped_ms = int(fperiod)

    # Launch side-effect of playing audio and play the frames 1 by 1
    play_audio_from_mp4(mp4_filepath, mp3_recv_tempdir)
    for frame in playing_frames:
        if end_convo_event.is_set():
            return True
        cv2.imshow(window_name, frame)
        cv2.waitKey(fperiod_clipped_ms)

    # Wait as little bit at the end since the audio might still be playing or generally we want to make sure things end in a clear fashion
    sample_end_period = 0.1
    start_time = time.time()
    while time.time() - start_time < wait_after_period:
        if end_convo_event.is_set():
            return True
        cv2.imshow(window_name, default_image)
        cv2.waitKey(1)

    # Signal the tts and listener to listen to us again
    playback_completion_event.set()
    return False


###


def convert2bmp(jpeg_file: Path, bmp_file: Path):
    # Open the JPEG file
    with Image.open(jpeg_file.as_posix()) as img:
        # Save as BMP
        img.save(bmp_file.as_posix(), "BMP")


### Main function ###
async def frame_player_proc_main_async(
    # IPC Communication
    tts_mp3_queue: multiprocessing.Queue,
    mp4_tempdir: Path,
    playback_completion_event: multiprocessing.Event,
    end_convo_event: multiprocessing.Event,
    # TODO(Adriano) make these arguments, also since we're going to have a lot of arguments, make arguments
    # YAML files so that it's easy to specify them!
    max_mp4s: int = 128,
    save_mp4s: bool = True,
    save_to_parent_dir=Path("~/Downloads").expanduser(),
    save_mp4s_zip_name: str = "mp4s-from-server",
) -> None:
    assert mp4_tempdir.exists()
    assert mp4_tempdir.is_dir()
    assert "REMOTE_IP" in os.environ

    # Alternative not supported for now :P
    assert save_mp4s

    # TODO(Adriano) less tempdirs
    with tempfile.TemporaryDirectory() as bmp_tempdir:
        print("--- (video client) --- converting heic to bmp")
        bmp_tempdir = Path(bmp_tempdir)
        bmp_filename = bmp_tempdir / "face.bmp"
        convert2bmp(Path(os.environ["FACE"]).expanduser().resolve(), bmp_filename)
        default_image = cv2.imread(bmp_filename.as_posix())
        assert default_image is not None

        mp4_file_handler = FileHandler(
            mp4_tempdir, max_mp4s, extension="mp4", save_zip_name=save_mp4s_zip_name
        )

        # Establish that mp4s are to be saved on exit
        def sigterm_handler(signum, frame):  # Note sure what the frame is? stack frame?
            mp4_file_handler.save_all_files_zip_to(save_to_parent_dir)

        signal.signal(signal.SIGTERM, sigterm_handler)

        # CV2 window
        window = "frame"

        # Ease of life
        wait_after_period = 1  # 1 sec

        # TODO(Adriano) don't use this ugly mp3 tempdir shit
        with tempfile.TemporaryDirectory() as mp3_recv_tempdir:
            mp3_recv_tempdir = Path(mp3_recv_tempdir)

            while True:
                mp3_filename, done = await await_tts_mp3_filename(
                    tts_mp3_queue,
                    end_convo_event,
                    bmp_filename,
                    # CV2 display
                    default_image,
                    window,
                )
                if done:
                    break
                assert mp3_filename is not None
                mp4_filename, done = await await_mp4_response(
                    end_convo_event,
                    bmp_filename,
                    mp3_filename,
                    mp4_file_handler,
                    # CV2 display
                    default_image,
                    window,
                )
                if done:
                    break
                done = await play_mp4_response(
                    end_convo_event,
                    mp4_filename,
                    playback_completion_event,
                    mp3_recv_tempdir,
                    # CV2 display
                    window,
                    default_image,
                    # Give it a little time to finish audio
                    wait_after_period,
                )
                if done:
                    break

            print("--- (video client) --- Done!")
            cv2.destroyAllWindows()
            print("--- (video client) --- Done destroying cv2 windows!")


###


### Wrapper for asyncio ###
def frame_player_proc_main(*args, **kwargs) -> None:
    return asyncio.run(frame_player_proc_main_async(*args, **kwargs))


###

# NOTE this is for Debug and you should use it to make sure the server is behaving nominally!
if __name__ == "__main__":
    print(" --- (video client MAIN) --- Running video client fetch test!")

    def play_mp4(mp4_file_path: str, tmpdir: Path) -> None:
        # Initialize Pygame for audio
        pygame.init()
        pygame.display.set_mode((1, 1))  # Minimal window

        # Load video file
        cap = cv2.VideoCapture(mp4_file_path)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Get framerate and choose a little it of time to just stop
        fps = cap.get(cv2.CAP_PROP_FPS)
        fperiod_ms = 1000 / fps
        fperiod_clipped_ms = int(fperiod_ms)
        wait_after_video_period_ms = 1000
        wait_after_video_period_clipped_ms = int(wait_after_video_period_ms)
        play_audio_from_mp4(mp4_file_path, tmpdir)
        for _ in range(int(num_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("frame", frame)
            if cv2.waitKey(fperiod_clipped_ms) & 0xFF in [ord("q"), 27]:
                break
        cv2.waitKey(wait_after_video_period_clipped_ms)

        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

    async def test_main():
        # Some samples
        contents = asyncio.ensure_future(
            recv_mp4(
                Path("speech-driven-animation-master/sample_face.bmp"),
                Path("audiocapture.mp3"),
            )
        )
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            mp4_filepath = tempdir / "result.mp4"
            mp4_bytes = await contents
            with open(mp4_filepath.as_posix(), "wb") as f:
                f.write(mp4_bytes)
            assert mp4_filepath.exists() and mp4_filepath.is_file()
            print(f" --- (video client MAIN) --- Done (in {mp4_filepath.as_posix()})")
            play_mp4(mp4_filepath.as_posix(), tempdir)

    asyncio.run(test_main())
