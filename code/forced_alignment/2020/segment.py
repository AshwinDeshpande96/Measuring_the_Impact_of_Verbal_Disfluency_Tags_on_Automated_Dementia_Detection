from global_var import *


def reformat_wav(audio):
    if system_type == "windows":
        audio = audio.replace('/mnt/f', "F:")
        audio = audio.replace('/', '\\')
    audio_file = AudioSegment.from_wav(audio)
    audio_file = audio_file.set_frame_rate(16000)
    audio_file.export(audio, format="wav")
