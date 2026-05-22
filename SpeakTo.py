import time
import speech_recognition as sr
import ollama
import sys
import os
import asyncio
import edge_tts
import pygame

class DollAI:
    def __init__(self):
        # Choose your local model here
        self.model_name = "qwen2.5:3b"
        
        # 1. Automate the model check and download
        self.ensure_model_exists()
        
        # Initialize Speech Recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize Audio Player (pygame Mixer is great for playing MP3s cleanly)
        pygame.mixer.init()
        
        # Configure Edge TTS Voice. "en-US-GuyNeural" or "en-GB-SoniaNeural" sound amazing!
        self.voice_character = "en-US-GuyNeural"
        
        # System instructions to give the doll a specific personality
        self.system_prompt = (
            "You are an amicale personal assistant who has been working for me for several years. "
            "Keep your answers brief, engaging, and professional (1-3 sentences maximum) "
            "as you are talking to a human friend or colleague. "
            "Do not write out reasoning blocks, use thinking mode, or use markdown text or asterisks."
        )

    def ensure_model_exists(self):
        """ Checks if the model is local; downloads it automatically if missing """
        print(f"[Checking local Ollama inventory for '{self.model_name}'...]")
        try:
            local_models = ollama.list()
            downloaded_names = [m['model'] for m in local_models.get('models', [])]
            
            if any(self.model_name in name for name in downloaded_names):
                print(f"[Brain Module Found: '{self.model_name}' is ready to go!]")
                return

            print(f"\n[Model '{self.model_name}' not found locally!]")
            print(f"[Downloading model files directly from Ollama registry now...]")
            
            current_status = ""
            for progress in ollama.pull(self.model_name, stream=True):
                status = progress.get('status', '')
                if status != current_status:
                    print(f" -> Status: {status}")
                    current_status = status
                    
            print(f"[Download and Initialization of '{self.model_name}' Complete!]\n")

        except Exception as e:
            print("\n[CRITICAL ERROR: Could not connect to the Ollama application.]")
            print(" -> Please make sure the Ollama application is running in the background of your laptop!")
            print(f" -> Error Details: {e}")
            sys.exit(1)

    def speak(self, text):
        """ Converts text to MP3 and plays it back smoothly over audio channels """
        print(f"\nDoll says: '{text}'")
        
        # Helper function to generate and play audio asynchronously
        async def generate_speech():
            output_filename = "response.mp3"
            communicate = edge_tts.Communicate(text, self.voice_character)
            await communicate.save(output_filename)
            
            # Play the generated audio file
            pygame.mixer.music.load(output_filename)
            pygame.mixer.music.play()
            
            # Block the script until the voice finishes speaking completely
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
            # Unload audio file so Windows/Mac releases the file lock
            pygame.mixer.music.unload()
            try:
                os.remove(output_filename)
            except Exception:
                pass

        # Execute the async speech generator cleanly within our standard synchronous loop
        asyncio.run(generate_speech())

    def listen(self):
        """ Listens to the microphone and converts it to text """
        with sr.Microphone() as source:
            print("\n[Listening... Speak into your microphone]")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                audio_data = self.recognizer.listen(source, timeout=25, phrase_time_limit=25)
                text_input = self.recognizer.recognize_google(audio_data)
                print(f"You said: '{text_input}'")
                return text_input
                
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                return None
            except sr.RequestError:
                print("[System Error: Check internet/network proxy connection]")
                return None

    def get_llm_response(self, user_text):
        """ Sends user text to the local Ollama LLM and returns the character response """
        print(f"[Thinking using {self.model_name}...]")
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': user_text}
            ])
            return response['message']['content']
        except Exception as e:
            return f"Brain module error. Could not reach Ollama. Error details: {e}"

    def run_loop(self):
        self.speak("System initialized. I am ready to talk!")
        
        while True:
            user_speech = self.listen()
            
            if user_speech and user_speech.strip():
                clean_speech = user_speech.strip()

                if "goodbye" in clean_speech.lower() or "quit" in clean_speech.lower():
                    self.speak("Goodbye for now, friend!")
                    break
                
                # 1. Save what you said to your diary log file
                with open("diary_log.txt", "a", encoding="utf-8") as file:
                    file.write(f"User: {clean_speech}\n")
                
                # 2. Think using the LLM
                ai_response = self.get_llm_response(clean_speech)
                
                # 3. Save what the AI said to the log file
                with open("diary_log.txt", "a", encoding="utf-8") as file:
                    file.write(f"Doll: {ai_response}\n\n")
                
                # 4. Speak the response aloud
                self.speak(ai_response)

if __name__ == "__main__":
    doll = DollAI()
    doll.run_loop()