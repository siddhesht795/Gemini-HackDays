# from dotenv import load_dotenv
# from elevenlabs.client import ElevenLabs
# from elevenlabs.play import play
# import os
# key = os.getenv("ELEVENLABS_API_KEY")
# print("KEY:", key)

# BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # hacksync2026/
# load_dotenv(os.path.join(BASE_DIR, ".env"))

# elevenlabs = ElevenLabs(
#   api_key=os.getenv("ELEVENLABS_API_KEY"),
# )

# audio = elevenlabs.text_to_speech.convert(
#     text="Yes, We're going to win this Hackathon",
#     voice_id="JBFqnCBsd6RMkjVDRZzb",
#     model_id="eleven_multilingual_v2",
#     output_format="mp3_44100_128",
# )

# print(type(audio))

# play(audio)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response
import os
from elevenlabs.client import ElevenLabs

router = APIRouter()

elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Voice IDs
VOICE_ID_MALE = "JBFqnCBsd6RMkjVDRZzb"
VOICE_ID_FEMALE = "KleDBQ7etYG6NMjnQ9Jw"

class TTSInput(BaseModel):
    text: str
    voice_id: str = None
    model_id: str = "eleven_multilingual_v2"
    gender: str = "male" # "male" or "female"

def get_voice_id(gender: str, explicit_voice_id: str = None) -> str:
    if explicit_voice_id:
        return explicit_voice_id
    if gender and gender.lower() == "female":
        return VOICE_ID_FEMALE
    return VOICE_ID_MALE

@router.post("/tts")
def tts(input: TTSInput):
    print(f"[TTS DEBUG] Endpoint called. Gender: {input.gender}, VoiceID input: {input.voice_id}")
    voice_id = get_voice_id(input.gender, input.voice_id)
    print(f"[TTS DEBUG] Selected Voice ID: {voice_id}")
    
    audio = elevenlabs.text_to_speech.convert(
        text=input.text,
        voice_id=voice_id,
        model_id=input.model_id,
        output_format="mp3_44100_128"
    )

    # audio is bytes-like iterable; convert to bytes
    audio_bytes = b"".join(audio)
    return Response(content=audio_bytes, media_type="audio/mpeg")

def tts_bytes(text, voice_id=None, model_id="eleven_multilingual_v2", gender="male"):
    print(f"[TTS_BYTES DEBUG] Called. Gender: {gender}, VoiceID input: {voice_id}")
    final_voice_id = get_voice_id(gender, voice_id)
    print(f"[TTS_BYTES DEBUG] Selected Voice ID: {final_voice_id}")
    
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id=final_voice_id,
        model_id=model_id,
        output_format="mp3_44100_128"
    )

    # audio is bytes-like iterable; convert to bytes
    audio_bytes = b"".join(audio)
    return audio_bytes