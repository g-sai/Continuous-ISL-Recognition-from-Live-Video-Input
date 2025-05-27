import os
from gtts import gTTS
from playsound import playsound
from dotenv import load_dotenv

load_dotenv()


def text_to_speech(text, language='en'):

    """
    Convert text to speech using gTTS and play the audio.
    
    Args:
        text (str): Text to convert to speech
    """

    tts = gTTS(text=text, lang=language, slow=False)
    audio_file = "output.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)