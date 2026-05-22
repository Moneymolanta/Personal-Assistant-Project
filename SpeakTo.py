import time
import speech_recognition as sr
import ollama
import sys
import os
import asyncio
import edge_tts
import pygame
import cv2
import threading


class DollAI:
    def __init__(self):
        # =========================
        # CAMERA SETUP
        # =========================
        self.cap = cv2.VideoCapture(0)
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.use_camera = True

        # Start camera thread
        threading.Thread(target=self.camera_loop, daemon=True).start()

        # =========================
        # MODEL SETUP
        # =========================
        self.chat_model = "qwen2.5:3b"
        self.vision_model = "moondream"

        self.ensure_model_exists()

        # =========================
        # SPEECH RECOGNITION
        # =========================
        self.recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        # =========================
        # AUDIO OUTPUT (TTS)
        # =========================
        pygame.mixer.init()
        self.voice_character = "en-US-GuyNeural"

        # =========================
        # SYSTEM PROMPT
        # =========================
        self.system_prompt = (
            "You are a helpful personal assistant. "
            "Keep responses short (1-3 sentences), natural, and conversational."
        )

    # =========================================================
    # MODEL CHECK
    # =========================================================
    def ensure_model_exists(self):
        print(f"[Checking models...]")

        try:
            local_models = ollama.list()
            downloaded = [m["model"] for m in local_models.get("models", [])]

            for model in [self.chat_model, self.vision_model]:
                if any(model in m for m in downloaded):
                    print(f"[OK] {model}")
                else:
                    print(f"[Downloading {model}]")
                    ollama.pull(model)

        except Exception as e:
            print("Ollama not running:", e)
            sys.exit(1)

    # =========================================================
    # CAMERA LOOP (FAST + SAFE)
    # =========================================================
    def camera_loop(self):
        while self.use_camera:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.last_frame = frame

            time.sleep(0.03)  # prevents CPU overload

    def get_latest_frame(self):
        with self.frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    # =========================================================
    # VISION (MOONDREAM)
    # =========================================================
    def describe_scene(self, question=None):
        frame = self.get_latest_frame()

        if frame is None:
            return "I cannot see anything right now."

        _, buffer = cv2.imencode(".jpg", frame)
        image_bytes = buffer.tobytes()

        prompt = question if question else "Describe what you see."

        response = ollama.chat(
            model=self.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes]
                }
            ]
        )

        return response["message"]["content"]

    # =========================================================
    # CHAT MODEL (QWEN)
    # =========================================================
    def get_llm_response(self, user_text):
        try:
            response = ollama.chat(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_text}
                ]
            )
            return response["message"]["content"]
        except Exception as e:
            return f"LLM error: {e}"

    # =========================================================
    # ROUTER (FAST - NO LLM NEEDED)
    # =========================================================
    def route_request(self, user_text):
        text = user_text.lower()

        vision_keywords = [
            "see", "look", "camera", "what is",
            "what's in", "describe", "image",
            "picture", "around me"
        ]

        if any(k in text for k in vision_keywords):
            print("[VISION MODE]")
            return self.describe_scene(user_text)

        print("[CHAT MODE]")
        return self.get_llm_response(user_text)

    # =========================================================
    # TTS
    # =========================================================
    def speak(self, text):
        print(f"\nDoll: {text}")

        async def run_tts():
            file = "response.mp3"

            tts = edge_tts.Communicate(text, self.voice_character)
            await tts.save(file)

            pygame.mixer.music.load(file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)

            pygame.mixer.music.unload()

            try:
                os.remove(file)
            except:
                pass

        asyncio.run(run_tts())

    # =========================================================
    # LISTEN
    # =========================================================
    def listen(self):
        with sr.Microphone() as source:
            print("\nListening...")
            try:
                audio = self.recognizer.listen(source, timeout=25, phrase_time_limit=25)
                text = self.recognizer.recognize_google(audio)
                print(f"You: {text}")
                return text
            except:
                return None

    # =========================================================
    # MAIN LOOP
    # =========================================================
    def run_loop(self):
        self.speak("System ready.")

        while True:
            user = self.listen()

            if not user:
                continue

            user = user.strip()

            if "quit" in user.lower() or "goodbye" in user.lower():
                self.speak("Goodbye.")
                break

            response = self.route_request(user)
            self.speak(response)

    # =========================================================
    # CLEANUP
    # =========================================================
    def cleanup(self):
        self.use_camera = False
        if self.cap:
            self.cap.release()


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    bot = DollAI()
    try:
        bot.run_loop()
    finally:
        bot.cleanup()