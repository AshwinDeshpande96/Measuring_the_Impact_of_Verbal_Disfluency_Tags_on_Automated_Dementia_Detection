import os

import pandas as pd

columns = ['speaker', 'mmse', 'dx', 'gold_transcript', 'asr_transcript', 'gold_transcript_w_pause',
           'asr_transcript_w_pause']
df = pd.read_csv('./stats.csv')
df['gold_transcript'] = ""
df['asr_transcript'] = ""
df['gold_transcript_w_pause'] = ""
df['asr_transcript_w_pause'] = ""
transcripts = os.listdir('./data/penn_files/text')
transcripts_w_pause = os.listdir('./data/penn_files/transcripts')


def read_file(transcript_file):
    with open(transcript_file) as fptr:
        file_text = fptr.read()
        fptr.close()
    return file_text


for transcript_file in transcripts:
    filename, _ = transcript_file.split('.')
    speaker, _, _, transcript_type = filename.split('_')
    transcript = read_file(f'./data/penn_files/text/{transcript_file}')
    df.loc[df['speaker'] == speaker, f'{transcript_type}_transcript'] += ". " + transcript

for transcript_file in transcripts_w_pause:
    filename, _ = transcript_file.split('.')
    speaker, _, _, transcript_type = filename.split('_')
    transcript = read_file(f'./data/penn_files/transcripts/{transcript_file}')
    df.loc[df['speaker'] == speaker, f'{transcript_type}_transcript_w_pause'] += ". " + transcript
df.to_pickle("data2020_pause.pickle")
