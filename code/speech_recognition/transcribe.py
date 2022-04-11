import os
import re

import pandas as pd
from pydub import AudioSegment

from google_asr import transcribe, sample_rate
from wav2vec import transcribe_wav2vec
from ..disfluency.remove_tags import remove_tags

directory = "/mnt/e/Research/ADReSSo2/2020/train/"


def read_data(transcript_filepath):
    with open(transcript_filepath, 'r') as read_fptr:
        text = read_fptr.read()
        read_fptr.close()
    text_lines = text.split("\n")
    return text_lines


def get_utterance(transcript_filepath):
    """
    Reads a transcript file ex: S001.cha
    Produces a list of utterances that belong to the participant along with its segment (start, end) in the audio clip
    :param transcript_filepath:
    :return: A list of [gold uterrance, segment] ex:
    line:  well there 's a mother standing there uh uh washing the dishes and the sink is overspilling
    segment: (4266, 13310)
    """
    lines = read_data(transcript_filepath)
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        segment = None
        # Find participant utterance
        if not line.startswith("*PAR:\t"):
            idx += 1
            continue
        # Join trailing utterances that belong to the participant
        forward_idx = idx
        while forward_idx < len(lines):
            forward_line = lines[forward_idx]
            if not line.endswith(forward_line):
                line += " " + forward_line
            if re.search(r"\d+\_\d+", forward_line):
                for seg in re.findall(r"\d+\_\d+", forward_line):
                    start, end = seg.split("_")
                    start = int(start)
                    end = int(end)
                    segment = (start, end)
                    break
                idx = forward_idx + 1
                break
            forward_idx += 1
        # Clean up text
        line = remove_tags(line)
        # Return Line and its segment
        yield line, segment


def get_audio_segment(speaker, segment, audio_filepath):
    """
    Creates new audio clip by slicing out the segment
    :param speaker: ex: S001
    :param segment: ex: (4266, 13310)
    :param audio_filepath: S001.wav
    :return: S001_4266_13310.wav
    """
    start, end = segment
    audio_file = AudioSegment.from_wav(audio_filepath)
    audio_file = audio_file.set_frame_rate(sample_rate)
    segment_path = './audio/{}_{}_{}.wav'.format(speaker, start, end)
    audio_segment = audio_file[start:end]
    audio_segment.export(segment_path, format="wav")
    return segment_path


def get_data(speaker, transcript_filepath, audio_filepath, transcribe=transcribe_wav2vec):
    """
    Takes as input transcript(CHAT protocol) filepath with a list of utterances
    Participant(subject) utterance segment is sliced out into a new smaller audio clip
    New Audio Clip is transcribed using ASR
    ASR output is added with gold transcript
    :param transcribe: Function that takes as input the path of the audio clip and returns ASR output
    :param speaker: Speaker ID ex: S001, S002
    :param transcript_filepath: Gold Transcript path ex: S001.cha, S002.cha
    :param audio_filepath: Audio path ex: S001.wav, S002.wav
    :return: A row of data consisting of
    utterance_segment ex: S001_4266_13310
    participant_utterance ex: well there 's a mother standing there uh uh washing the dishes and the sink is overspilling
    asr_utterance ex: well there 's another standing there washing the dishes in the sink is over spelling
    segment_filepath ex: ./audio/S001_4266_13310.wav
    """
    for participant_utterance, segment in get_utterance(transcript_filepath):
        utt_id = f"{speaker}_{segment[0]}_{segment[1]}"
        # Create segment audio clip
        segment_filepath = get_audio_segment(speaker, segment, audio_filepath)
        # Transcribe segment
        asr_utterance = transcribe(segment_filepath)
        yield [utt_id, participant_utterance, asr_utterance, segment_filepath]


def main():
    """
    Transcribe using ASR of choice
    Default ASR: Google ASR
    Requires
    1. GOOGLE_APPLICATION_CREDENTIALS obtained as a part of GCP API
    2. sample_rate: 16000. Do not change
    3. directory: ADReSS data structured as:
    ADDRESS_DIRECTORY ----- train ----- Full_wave_enhanced_audio ----- cc ----- [S001.wav, S002.wav, ...]
                       |           |                               |
                       |           |                               --- cd ----- [S079.wav, S080.wav, ...]
                       |           |
                       |           ---- transcription ----- Control ----- [S001.cha, S002.cha, ...]
                       |                                |
                       |                                --- Dementia ____ [S079.cha, S080.cha, ...]
                       |
                       |
                       ---- test ----- Full_wave_enhanced_audio ---- [S150.wav, S151.wav, ...]
                                  |
                                  ---- transcription ---- [S150.cha, S151.cha, ...]
    :return:
    """
    # Create a empty dataframe that maps gold utterance to asr utterance
    columns = ['utt_id', 'gold_utterance', 'asr_utterance', 'audio_filepath']
    df = pd.DataFrame(columns=columns)
    # create a row for each utterance
    # Input - speaker audio and transcript (S001.wav, S001.cha)
    # Output - List of [utterance, gold transcript, asr transcript]
    for cat1, cat2 in zip(["cc", "cd"], ["Control", "Dementia"]):
        cat_path = os.path.join(directory, 'transcription', cat2)
        audio_path = os.path.join(directory, 'Full_wave_enhanced_audio', cat1)
        for file in os.listdir(cat_path):
            transcript_filepath = os.path.join(cat_path, file)
            speaker, _ = file.split('.')
            audio_filepath = os.path.join(audio_path, f"{speaker}.wav")
            # get_data() produces a list of utterances given speaker audio and transcript
            for data in get_data(speaker, transcript_filepath, audio_filepath, transcribe):
                row = dict(zip(columns, data))
                df = df.append(row, ignore_index=True)
    df.to_pickle('adresso2020_transcripts.pickle')


if __name__ == '__main__':
    main()
