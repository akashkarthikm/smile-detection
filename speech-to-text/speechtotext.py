#seperate audio file included for testing

import speech_recognition as sr

def speech_to_text_from_file(file_path):
    recognizer = sr.Recognizer()

    # Use the specified audio file as the source
    with sr.AudioFile(file_path) as source:
        print(f"Processing audio file: {file_path}")
        try:
            audio_data = recognizer.listen(source)  # Records the entire audio file
            text = recognizer.recognize_google(audio_data)
            print("Text from audio file:", text)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    file_path = "./harvard.wav"  # Replace with the path to your audio file
    speech_to_text_from_file(file_path)
