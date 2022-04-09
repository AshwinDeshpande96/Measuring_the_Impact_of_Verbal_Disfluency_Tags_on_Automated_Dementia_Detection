from align import *
from segment import *


class Start:
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.DataFrame()

    def read_df(self):
        df = pd.read_csv(self.filename, usecols=['speaker', 'segments', 'audio', 'transcript'])
        df.segments = df.segments.apply(lambda x: literal_eval(x))
        df.transcript = df.transcript.apply(lambda x: literal_eval(x))
        self.data = df

    def get_alignments(self):
        for (index, (speaker, segments, audio, transcript)) in self.data[['speaker',
                                                                          'segments',
                                                                          'audio',
                                                                          'transcript']].iterrows():
            for audio_file, transcript in zip(segment_wav(speaker, segments, audio), transcript):
                align_using_penn(audio_file, transcript)
