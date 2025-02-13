import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()

def SpeechService():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Request error: {e}")
        return None

def speak_text(text):
    engine.say(text)
    engine.runAndWait()