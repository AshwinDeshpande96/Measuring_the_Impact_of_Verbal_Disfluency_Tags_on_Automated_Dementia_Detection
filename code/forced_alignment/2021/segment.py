from global_var import *


def segment_wav(speaker, segments, audio):
    segmented_audio = []
    if system_type == "windows":
        audio = audio.replace('/mnt/f', "F:")
        audio = audio.replace('/', '\\')
    audio_file = AudioSegment.from_wav(audio)
    audio_file = audio_file.set_frame_rate(16000)
    for begin, end in segments:
        segment_path = os.path.join("data", "audio", "{}_{}_{}.wav".format(speaker, begin, end))
        audio_segment = audio_file[begin:end]
        audio_segment.export(segment_path, format="wav")
        segmented_audio.append(segment_path)
    return segmented_audio
