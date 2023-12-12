# NOTE that you must also have REMOTE_IP as an environment variable pointing to the IP or generally hostname of the server that's running the ML backend as well as
# another environment variable with the value OPENAI_API_KEY set to your OpenAI API key as well as an environment variable for FACE which is pointing to the face jpg or bmp or whatever...

# Adds .en if you pick english; not sure where these come from (just using whisper library); I think it downloads OSS models to some directory on your computer (maybe ~/.cache/whisper?)
DEFAULT_MODEL = "tiny"
DEFAULT_NON_ENGLISH = False

# Linux specific; not my code
DEFAULT_MICROPHONE_LINUX = "pulse"

# Not my code
DEFAULT_ENERGY_THRESHOLD: int = 1000

# Note all time units are in seconds
DEFAULT_RECORD_TIMEOUT: float = 2
DEFAULT_PHRASE_TIMEOUT: float = 3
DEFAULT_INITIAL_PHRASE_TIMEOUT: float = 9
DEFAULT_SLEEP_AFTER_RESPONSE: float = 0.1

# How many conversational steps to take; 0 denotes infinite
DEFAULT_NUM_CONVERSATIONAL_STEPS: int = 0
