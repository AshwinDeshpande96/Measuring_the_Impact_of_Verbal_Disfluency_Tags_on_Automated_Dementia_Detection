from global_var import *

word_aligned_files = os.listdir(os.path.join('..', 'data', "penn_files", 'word_out'))
audio_files = os.listdir(os.path.join('..', 'data', 'audio'))

for file in r.sample(audio_files, len(audio_files)):
    audio = os.path.join('..', 'data', 'audio', file)
    segment_id = audio.split(directory_seperator)[-1].split('.')[0]
    speaker_id = segment_id.split('_')[0]
    speaker_pause = {'0-500': None, '500-1000': None, '1000-2000': None, '2000': None}
    speaker_transcript = []
    aligned_filename = segment_id + ".pickle"
    if not aligned_filename in word_aligned_files:
        continue
    audio_file = AudioSegment.from_wav(audio)
    with open(os.path.join('..', 'data', "penn_files", 'word_out', aligned_filename), 'rb') as f:
        data = pickle.load(f)
    if data and data[0][0] == 'sp':
        data = data[1:]
    if data and data[-1][0] == 'sp':
        data = data[:-1]
    for (word, begin, end) in data:
        if word != "sp":
            speaker_transcript.append(word)
            continue
        begin = float(begin) * 1000
        end = float(end) * 1000
        duration = end - begin
        if duration < 50:
            continue
        elif duration < 500:
            if speaker_pause['0-500'] is None:
                speaker_pause['0-500'] = 0
            speaker_pause['0-500'] += 1
            speaker_transcript.append('[P1]')
        elif duration < 1000:
            if speaker_pause['500-1000'] is None:
                speaker_pause['500-1000'] = 0
            speaker_pause['500-1000'] += 1
            speaker_transcript.append('[P2]')
        elif duration < 2000:
            if speaker_pause['1000-2000'] is None:
                speaker_pause['1000-2000'] = 0
            speaker_pause['1000-2000'] += 1
            speaker_transcript.append('[P2]')
        else:
            if speaker_pause['2000'] is None:
                speaker_pause['2000'] = 0
            speaker_pause['2000'] += 1
            speaker_transcript.append('[P3]')
        # play(audio_file)
    pause_file = os.path.join("data", "penn_files", "pause_features", "{}.pickle".format(segment_id))
    pickle.dump(speaker_pause, open(pause_file, 'wb'))
    transcript_file = os.path.join("data", "penn_files", "transcripts", "{}.txt".format(segment_id))
    with open(transcript_file, 'w') as fptr:
        fptr.write(" ".join(speaker_transcript))
