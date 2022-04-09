from global_var import *

aligned_files = os.listdir(os.path.join('..', 'data', "aeneas_files", 'out'))
audio_files = os.listdir(os.path.join('..', 'data', "aeneas_files", 'audio'))
for file in r.sample(audio_files, len(audio_files)):
    audio = os.path.join('..', 'data', "aeneas_files", 'audio', file)
    segment_id = audio.split(directory_seperator)[-1].split('.')[0]
    aligned_filename = segment_id + ".json"
    if not aligned_filename in aligned_files:
        continue
    audio_file = AudioSegment.from_wav(audio)
    with open(os.path.join('..', 'data', "aeneas_files", 'out', aligned_filename)) as f:
        data = json.load(f)
    for word_fragment in data['fragments']:
        begin, children, end, wid, language, lines = word_fragment.values()
        begin = float(begin) * 1000
        end = float(end) * 1000
        word_segment = audio_file[begin:end]
        print(lines[0])
        play(word_segment)
        input()
