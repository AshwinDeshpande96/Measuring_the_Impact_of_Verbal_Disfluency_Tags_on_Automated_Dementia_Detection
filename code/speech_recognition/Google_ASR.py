import io
import os

from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_CREDENTIALS.json"

sample_rate = 16000


def transcribe(speech_file):
    full_result = ""
    # Transcribe the given audio file

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                                      sample_rate_hertz=sample_rate,
                                      language_code="en-US",
                                      enable_automatic_punctuation=True,
                                      use_enhanced=True,
                                      model="phone_call",
                                      )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        full_result += result.alternatives[0].transcript
        # print(u"Transcript: {}".format(result.alternatives[0].transcript))
    return full_result
