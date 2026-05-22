import io
import os
import sys
import pyaudio
import wave
import numpy as np
from faster_whisper import WhisperModel
import pyttsx3

class LocalFastDoll:
    def __init__(self):
        print("[Loading local Audio Model...]")
        # Using tiny.en since it already downloaded successfully on your machine
        self.model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        print("[Model Loaded Successfully]")

        # Initialize Text-to-Speech
        self.tts_engine = pyttsx3.init()
        rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', rate - 30)

        # Audio Recording Configurations
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  
        self.CHUNK = 1024
        self.p = pyaudio.PyAudio()

    def speak(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen_and_transcribe(self):
        print("\n[Listening... Speak into mic]")
        
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS,
                             rate=self.RATE, input=True,
                             frames_per_buffer=self.CHUNK)
        
        frames = []
        silence_threshold =25  # Adjust this threshold based on your microphone sensitivity and environment  
        silent_chunks = 0
        has_spoken = False
        
        while True:
            data = stream.read(self.CHUNK)
            frames.append(data)
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            amplitude = np.abs(audio_data).mean()
            
            if amplitude > silence_threshold:
                has_spoken = True
                silent_chunks = 0
            else:
                if has_spoken:
                    silent_chunks += 1
            
            # Stop recording after ~1 second of silence following speech
            if has_spoken and silent_chunks > (self.RATE / self.CHUNK * 1):
                print("[End of speech detected]")
                break
                
        stream.stop_stream()
        stream.close()
        
        audio_buffer = io.BytesIO()
        wf = wave.open(audio_buffer, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        audio_buffer.seek(0)
        
        print("[Transcribing locally...]")
        segments, info = self.model.transcribe(audio_buffer, beam_size=1)
        text_input = "".join([segment.text for segment in segments]).strip()
        print(f"You said: '{text_input}'")
        return text_input

    def run(self):
        """The main loop controlling the doll's state"""

        while True:
            try:
                user_text = self.listen_and_transcribe()
                if user_text:
                    if "goodbye" in user_text.lower() or "quit" in user_text.lower():
                        self.speak("Powering down.")
                        break
                    
                    # Echo response placeholder
                    self.speak(f"Doll: {user_text}")
            except KeyboardInterrupt:
                break
        
        # Clean up audio hardware
        self.p.terminate()

if __name__ == "__main__":
    print('Program Ready...')
    doll = LocalFastDoll()
    try:
        doll.run()
    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        # Force Python to delete the tts engine safely before exiting 
        # to prevent the Python 3.13 DriverProxy crash warning.
        if hasattr(doll, 'tts_engine'):
            del doll.tts_engine