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

#heyyyyyyyy
#ok so what I just did was add a new function called attention_check that runs in a loop in the background. It checks for motion by comparing the current camera frame to the previous one. If it detects significant motion, it logs an event in memory. This way, the doll can be more responsive to changes in its environment without needing to wait for a user query.
#and wtv else the shit says up there, I didn't erite it
#anyway, I need to figure out how the model can have visual context without outright prompting
#or with any keywords
#chat's saying I need to change the intital model prompt
#and update to a 3class sytem to include the memory of the system
#look at chat's most recent text to get context
#gn
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
        self.model_name = "qwen2.5vl:3b"
        self.ensure_model_exists()

        # =========================
        # Memory Setup
        # =========================
        self.memory = {
            "last_scene": None,
            "last_objects": [],
            "last_update_time": None,
            "events": []   # small event log
        }

        # =========================
        # Speech Rec
        # =========================
        self.recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        # =========================
        # Audio Output (tts)
        # =========================
        pygame.mixer.init()
        self.voice_character = "en-US-AndrewNeural"

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

            for model in [self.model_name]:
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
    def attention_loop(self):
            while self.use_camera:
                self.attention_check()
                time.sleep(0.2)

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
    # CHAT MODEL (QWEN)
    # =========================================================
    def generate_response(
        self,
        user_text,
        include_image=False,
        include_memory=False
    ):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        memory_context = ""

        if include_memory and self.memory["events"]:
            recent_events = self.memory["events"][-5:]

            memory_context = "\n".join(
                [str(event) for event in recent_events]
            )

        prompt = f"""
    User message:
    {user_text}
    """

        if include_memory:
            prompt += f"""

    Recent memory/events:
    {memory_context}
    """

        message = {
            "role": "user",
            "content": prompt
        }

        if include_image:
            frame = self.get_latest_frame()

            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame)
                image_bytes = buffer.tobytes()

                message["images"] = [image_bytes]

        messages.append(message)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )

            result = response["message"]["content"]

            self.memory["events"].append({
                "time": time.time(),
                "event": "assistant_response",
                "data": result
            })

            if len(self.memory["events"]) > 20:
                self.memory["events"].pop(0)

            return result

        except Exception as e:
            return f"LLM error: {e}"
        
    # =========================================================
    # Attention Check 
    # =========================================================

    def attention_check(self):
        frame = self.get_latest_frame()
        if frame is None:
            return

        # lightweight trigger: frame change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if not hasattr(self, "prev_frame"):
            self.prev_frame = gray
            return

        diff = cv2.absdiff(self.prev_frame, gray)
        score = diff.mean()

        self.prev_frame = gray

        if score > 10:  # motion threshold
            self.memory["events"].append({
                "time": time.time(),
                "event": "motion_detected",
                "score": float(score)
            })

    def answer_from_memory(self, user_text):
        if not self.memory.get("events"):
            return "I don't have any stored events yet."

        recent = self.memory["events"][-5:]

        context = "\n".join(
            [f"- {e['event']}: {e['data']}" for e in recent]
        )

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Use the memory context to answer."
                },
                {
                    "role": "user",
                    "content": f"""
    Memory:
    {context}

    Question:
    {user_text}
    """
                }
            ]
        )

        return response["message"]["content"]

    # =========================================================
    # ROUTER
    # =========================================================
    def route_request(self, user_text):

        router_prompt = f"""
    You are a routing system for a robot assistant.

    Determine which context sources are needed.

    Possible context sources:
    - IMAGE = requires current camera view
    - MEMORY = requires recent event memory

    Examples:

    "What color is the cat?"
    -> IMAGE

    "What did I just do?"
    -> IMAGE,MEMORY

    "What were you looking at earlier?"
    -> MEMORY

    "Tell me a joke."
    -> NONE

    User message:
    {user_text}

    Respond ONLY as comma-separated labels.

    Examples:
    IMAGE
    MEMORY
    IMAGE,MEMORY
    NONE
    """

        try:
            decision = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": router_prompt}
                ]
            )["message"]["content"].strip().upper()

        except Exception:
            decision = "NONE"

        include_image = "IMAGE" in decision
        include_memory = "MEMORY" in decision

        print(f"[ROUTER] image={include_image}, memory={include_memory}")

        return self.generate_response(
            user_text,
            include_image=include_image,
            include_memory=include_memory
        )

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