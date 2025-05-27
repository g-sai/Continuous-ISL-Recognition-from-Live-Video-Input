import os
from deepgram import DeepgramClient, SpeakOptions
import pygame
from dotenv import load_dotenv

load_dotenv()



def text_to_speech(text,output_file='response.wav'):

    """
    Convert text to speech using Deepgram and play the audio.
    
    Args:
        text (str): Text to convert to speech
        output_file (str, optional): Path to save audio file. Defaults to 'response.wav'
    """

    try:
        deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        deepgram = DeepgramClient(api_key=deepgram_api_key)
        
        options = SpeakOptions(
            model="aura-luna-en",  #You can change the voice model
            encoding="linear16",
            container="wav"
        )
        speak_options = {"text": text}
        
        response = deepgram.speak.v("1").save(output_file, speak_options, options)
        
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.quit()
        os.remove(output_file)
    
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

