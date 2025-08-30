import os
import math
import threading
from datetime import datetime

from vpython import sphere, vector, color, rate, scene
import speech_recognition as sr
import pyttsx3


try:
    import groq
except ImportError:
    raise ImportError("groq SDK missing. Run: pip install groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. Obtain key at console.groq.com and set env var.")

MODEL_ID = os.getenv("GROQ_MODEL", "llama3-70b-8192")
groq_client = groq.Groq(api_key=GROQ_API_KEY)
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Neo")


# Text‑to‑Speech

engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text: str):
    print(f"{ASSISTANT_NAME}: {text}")
    engine.say(text)
    engine.runAndWait()


# Groq Chat Completions helper


def ask_groq(prompt: str) -> str:
    """Send a prompt to Groq and return the assistant's reply."""
    response = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": f"You are {ASSISTANT_NAME}, a helpful virtual assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# 3‑D Visual Assistant (VPython)

scene.title = "3D Virtual Assistant"
scene.background = color.black
assistant_ball = sphere(pos=vector(0, 0, 0), radius=1, color=color.cyan, emissive=True)

def animate():
    angle = 0
    while True:
        rate(60)
        angle += 0.05
        assistant_ball.color = vector(abs(math.sin(angle)), 0.8, 1 - abs(math.cos(angle)))
        assistant_ball.pos = vector(0, 0.2 * math.sin(angle), 0)

animation_thread = threading.Thread(target=animate, daemon=True)
animation_thread.start()


# Voice Loop with Groq fallback


def listen_and_respond():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        speak(f"Hello! I am {ASSISTANT_NAME}, your 3‑D assistant. How can I help you?")

        while True:
            print("Listening… (say 'exit' to quit)")
            try:
                audio = recognizer.listen(source, timeout=10)
                command = recognizer.recognize_google(audio).lower()
                print("You:", command)

                # Local quick commands
                if any(x in command for x in ("exit", "quit")):
                    speak("Goodbye!")
                    break
                if "your name" in command:
                    speak(f"I am {ASSISTANT_NAME}, your 3‑D virtual assistant.")
                    continue
                if "time" in command:
                    now = datetime.now().strftime("%H:%M:%S")
                    speak(f"The current time is {now}")
                    continue

                # Anything else → Groq LLM
                reply = ask_groq(command)
                speak(reply)

            except sr.WaitTimeoutError:
                speak("No input detected.")
            except sr.UnknownValueError:
                speak("Sorry, I couldn't understand.")
            except Exception as e:
                speak(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    listen_and_respond()

