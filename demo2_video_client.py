# TODO(Adriano) how can we improve the whole situation with the audio being out of sync with the video?

import requests
import cv2
import pygame
import os
import tempfile
import aiohttp
import asyncio
import multiprocessing
import signal
import numpy as np
from moviepy.editor import VideoFileClip
from pathlib import Path

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
    video = VideoFileClip(mp4_filepath)
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
    assert "FACE" in os.environ and os.environ["FACE"].endswith(".bmp")

    # Alternative not supported for now :P
    assert save_mp4s

    default_image = cv2.imread(os.environ["FACE"])
    assert default_image is not None

    mp4_file_handler = FileHandler(
        mp4_tempdir, max_mp4s, extension="mp4", zip_name=save_mp4s_zip_name
    )

    # Establish that mp4s are to be saved on exit
    def sigterm_handler(signum, frame):  # Note sure what the frame is? stack frame?
        mp4_file_handler.save_all_files_zip_to(save_to_parent_dir)

    signal.signal(signal.SIGTERM, sigterm_handler)

    # NOTE that the fps may change if we recieve an mp4 with a different fps
    default_fps = 60
    default_fperiod_ms = 1000 / default_fps
    default_fperiod_clipped_ms = int(default_fperiod_ms)
    fperiod_clipped_ms = default_fperiod_clipped_ms
    wait_after_video_period_ms = 3000  # Usually deals with any overflow from the error TODO(Adriano) fix this maybe?
    wait_after_video_period_clipped_ms = int(wait_after_video_period_ms)

    window_name = "frame"  # Window management # XXX use this!

    awaiting_mp4_response = False  # State management
    awaiting_mp4_promise = None  # State management

    playing_mp4_response = False
    playing_frame_number = None
    playing_frames = None
    # TODO(Adriano) don't use this ugly mp3 tempdir shit
    with tempfile.TemporaryDirectory() as mp3_recv_tempdir:
        mp3_recv_tempdir = Path(mp3_recv_tempdir)
        while True:
            if end_convo_event.is_set():  # Done
                break  # If we are signalled to end the whole proc then end
            elif (
                awaiting_mp4_response
            ):  # Default image frame or first frame (technically superfluous)
                assert not playing_mp4_response
                assert awaiting_mp4_promise is not None
                if awaiting_mp4_promise.done():
                    print("--- (video client) --- recv response, write file")
                    mp4_filename = write_mp4_to_tempfile(
                        awaiting_mp4_promise.result(), mp4_file_handler
                    )
                    awaiting_mp4_response = False
                    playing_mp4_response = True
                    awaiting_mp4_promise = None
                    playing_frame_number = 0
                    playing_frames, fps = get_frames_from_mp4(mp4_filename)
                    fperiod = 1000 / fps
                    fperiod_clipped_ms = int(fperiod)  # NOTE we overwrite the fperiod!

                    # TODO(Adraino) fix unimportant bug where we play twice
                    frame = playing_frames[playing_frame_number]

                    # NOTE side-effect to start playing audio! (This is the reason we had to switch to the fps
                    # specifically from the mp4... not sure how to better coordinate though)
                    play_audio_from_mp4(mp4_filename, mp3_recv_tempdir)

                    print("--- (video client) --- awaital done!")
                else:
                    cv2.imshow(window_name, default_image)
                    k = cv2.waitKey(fperiod_clipped_ms)
            elif playing_mp4_response:  # Next frame to play
                assert playing_frame_number < len(playing_frames)
                frame = playing_frames[playing_frame_number]
                playing_frame_number += 1
                if playing_frame_number == len(playing_frames):
                    playing_mp4_response = False
                    playing_frame_number = None
                    playing_frames = None
                    fperiod_clipped_ms = default_fperiod_clipped_ms
                    print("--- (video client) --- playback compelete")
                    # TODO(Adriano) not really clean to do this HERE
                    # wait those 3 sec for done speaking
                    cv2.waitKey(wait_after_video_period_clipped_ms)
                    playback_completion_event.set()
            else:  # Default image frame
                frame = default_image
                try:
                    # If a new MP3 is available, we are now going to try to play it
                    mp3_filename = tts_mp3_queue.get(timeout=0.1)
                    assert Path(mp3_filename).exists()
                    bmp_filename = os.environ["FACE"]
                    assert Path(bmp_filename).exists()
                    assert not awaiting_mp4_response
                    assert not playing_mp4_response
                    assert awaiting_mp4_promise is None

                    print(" --- (video client) --- commencing mp4 awaital")
                    awaiting_mp4_response = True
                    awaiting_mp4_promise = asyncio.ensure_future(
                        recv_mp4(
                            Path(bmp_filename),
                            Path(mp3_filename),
                        )
                    )
                except multiprocessing.queues.Empty:
                    pass  # Nothing happening here
            assert frame is not None
            cv2.imshow(window_name, frame)
            k = cv2.waitKey(fperiod_clipped_ms)
            # Exit on q or 27
            if k & 0xFF == ord("q") or k == 27:
                # NOTE you shouldn't actually use this though, it doesn't really work :P
                # TODO(Adriano) add support for this (if you do, you'll just be doing TTS
                # side and so it'll block forever)
                break

        cv2.destroyAllWindows()
        print("--- (video client) --- Done destroying cv2 windows!")


# Wrapper for asyncio
def frame_player_proc_main(*args, **kwargs) -> None:
    return asyncio.run(frame_player_proc_main_async(*args, **kwargs))


# NOTE this is for Debug and you should use it to make sure the server is behaving nominally!
if __name__ == "__main__":

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
            print(f"Done (in {mp4_filepath.as_posix()})")
            play_mp4(mp4_filepath.as_posix())

    asyncio.run(test_main())
