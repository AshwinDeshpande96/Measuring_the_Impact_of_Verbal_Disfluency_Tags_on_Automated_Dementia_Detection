from align import *
from segment import *


class Start:
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.DataFrame()

    def read_df(self):
        self.data = pd.read_pickle(self.filename)

    def get_alignments(self):
        for (index, (speaker, segments,
                     audio, gold_transcript,
                     asr_transcript)) in self.data[['speaker',
                                                    'segments',
                                                    'audio',
                                                    'gold_transcript',
                                                    'asr_transcript']].iterrows():
            for audio_file, transcript in zip(audio, gold_transcript):
                align_using_penn(audio_file, transcript, "gold")
            for audio_file, transcript in zip(audio, asr_transcript):
                align_using_penn(audio_file, transcript, "asr")
