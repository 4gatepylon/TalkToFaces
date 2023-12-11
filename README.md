# MeruCharacterAIDemo
A Series of demos that do character.ai-like functionality i which you can talk to a character. These are not meant for production.


## How to set up (on Mac)
We will use a speech recognition library that is already used using open source tools: https://github.com/Uberi/speech_recognition#readme. Specifically, we'll be using whisper. Make sure to follow all the tutorials there and here: https://github.com/openai/whisper (to get set up).

Make sure you are at `Python 3.10.9` or above.

### Testing FFMPEG (Not Real Time)
1. Find the proper input(s) (devices) to use for you: `ffmpeg -f avfoundation -list_devices true -i ""`
2. Capture from the proper device (assuming only audio and using device 1): `ffmpeg -f avfoundation -i ":1" -t 10 audiocapture.mp3`. Note that AVFoundation seems to be a Mac thing: https://developer.apple.com/av-foundation/. If you are on another platform, consider looking up a different tutorial.

### Setting up Real Time Speech to Text
TODO

### (STReTS) Speech to text to response to text to speech Demo
TODO. Make sure you have your ChatGPT key properly in the `.env`

Make sure to run with `export $(cat .env) && python3 demo1.py --model tiny --num-conversational-steps 0`

## Core Capabilties
1. `Done`; Transform speech into text. Be able to see on a terminal or equivalent the text output in real time, or pipe it (either literally or in a library) to a user system.
2. `Done`; Be able to use OpenAI ChatGPT to respond to any sort of prompt in a conversational format.
3. `Done`; Be able to use text to speech to have a voice conversation with ChatGPT.
4. Be able to animate a character of any kind, whether 2D, 3D or anything else, speaking with the text to speech, such that the lip movements are more or less in sync and look OK.
5. Be able to animate a 3D character of any kind to do the same as (4), or at least a 2D character that looks good. There are many things to explore after this, such as deepfaking (etc).
6. Be able to turn this into a full end to end web (or local application) experience that is monetizeable. Try to go out and do some sales and get some traction. It may be smart to read more about, and ideally talk with people in, Character AI and its competitors. Maybe I can scrape their website to gather information about their characters. It may also be important to try to start to get real time streaming or other such functionality to work well.

## More stuff
A lot of different models that I am trying to copy from research are here: https://github.com/4gatepylon/FaceFormer. At some point all will be merged into the monorepo.