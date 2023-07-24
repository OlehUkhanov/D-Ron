import openai
import asyncio
import re
import whisper
import boto3
import pydub
from pydub import playback
import speech_recognition as sr
from EdgeGPT import Chatbot, ConversationStyle
from numba import jit
import os

# Initialize the OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a recognizer object and wake word variables
recognizer = sr.Recognizer()

def synthesize_speech(text, output_filename):
  polly = boto3.client('polly', region_name='us-west-2')
  response = polly.synthesize_speech(
    Text=text,
    OutputFormat='mp3',
    VoiceId='Salli',
    Engine='neural'
  )

  with open(output_filename, 'wb') as f:
    f.write(response['AudioStream'].read())

def play_audio(file):
  sound = pydub.AudioSegment.from_file(file, format="mp3")
  playback.play(sound)

async def main():
  try:
    model = whisper.load_model("base.en")
    result = model.transcribe("audio.wav", fp16=False)
    phrase = result["text"]
    print(f"You said: {phrase}")
  except Exception as e:
    print("Error transcribing audio: {0}".format(e))

if __name__ == "__main__":
  asyncio.run(main())
