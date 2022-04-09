from global_var import *

word_aligned_files = os.listdir(os.path.join('.', 'data', "penn_files", 'word_out'))
audio_files = os.listdir(os.path.join('.', 'data', 'audio'))


def compute_vad_percent(filename):
    data = bob.io.audio.reader(filename)
    vad = bob.kaldi.compute_dnn_vad(data.load()[0], data.rate)
    ones = np.count_nonzero(vad)
    return ones / len(vad)


def is_filled(word_segment, segment_id, word):
    filepath = './tmp/{}_{}.wav'.format(segment_id, word)
    word_segment.export(filepath, format="wav")
    rate, audData = scipy.io.wavfile.read(filepath)
    energy = np.sum(audData.astype(float) ** 2)
    vad = compute_vad_percent(filepath)
    return True if vad > 0.5 else False


for file in r.sample(audio_files, len(audio_files)):
    audio = os.path.join('.', 'data', 'audio', file)
    segment_id = audio.split(directory_seperator)[-1].split('.')[0]
    aligned_filename = segment_id + ".pickle"
    if not aligned_filename in word_aligned_files:
        continue
    audio_file = AudioSegment.from_wav(audio)
    with open(os.path.join('.', 'data', "penn_files", 'word_out', aligned_filename), 'rb') as f:
        data = pickle.load(f)
    for (word, begin, end) in data:
        if word != "sp":
            continue
        begin = float(begin) * 1000
        end = float(end) * 1000
        word_segment = audio_file[begin:end]
        if is_filled(word_segment, segment_id, word):
            print(word)
            # play(word_segment)
            input()
    exit(0)
