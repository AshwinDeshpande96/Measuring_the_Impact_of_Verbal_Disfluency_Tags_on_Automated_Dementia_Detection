import numpy as np
import pandas as pd

from repetition.gold import remove_repetition
from retracing.gold import remove_retracing
from util import remove_tags, clean_text, count_words, get_percentage


def merge_speaker_utterances(data):
    data_final = {}
    for (idx, (speaker,
               transcript_without_tags,
               transcript_without_repetition,
               transcript_without_retracing,
               dx,
               mmse)) in data[['speaker',
                               'transcript_without_tags',
                               'transcript_without_repetition',
                               'transcript_without_retracing',
                               'dx',
                               'mmse']].iterrows():
        if speaker not in data_final:
            data_final[speaker] = (speaker,
                                   transcript_without_tags,
                                   transcript_without_repetition,
                                   transcript_without_retracing,
                                   dx,
                                   mmse)
        else:
            _transcript_without_tags = data_final[speaker][1]
            _transcript_without_repetition = data_final[speaker][2]
            _transcript_without_retracing = data_final[speaker][3]
            if transcript_without_tags is not None:
                _transcript_without_tags += " " + transcript_without_tags
            if transcript_without_repetition is not None:
                _transcript_without_repetition += " " + transcript_without_repetition
            if transcript_without_retracing is not None:
                _transcript_without_retracing += " " + transcript_without_retracing
            data_final[speaker] = (speaker,
                                   _transcript_without_tags,
                                   _transcript_without_repetition,
                                   _transcript_without_retracing,
                                   dx,
                                   mmse)

    data_final = pd.DataFrame(list(data_final.values()), columns=['speaker',
                                                                  'transcript_without_tags',
                                                                  'transcript_without_repetition',
                                                                  'transcript_without_retracing',
                                                                  'dx',
                                                                  'mmse'])
    return data_final


def main():
    # Import MMSE score for each speaker
    df = pd.read_csv('data.csv', usecols=['ID', 'mmse'])
    id_to_mmse = df.set_index('ID')['mmse'].to_dict()
    # Import Data
    data = pd.read_pickle('../transcript_with_disfluency_parse_train.pickle')
    # 1. Add MMSE score to each utterance
    data['mmse'] = data.speaker.apply(lambda x: id_to_mmse.get(x, None))
    # 2. Create a column with no disfluency markers
    data['transcript_without_tags'] = data.transcript_with_tags.apply(lambda x: remove_tags(x))
    data['transcript_without_tags'] = data.transcript_without_tags.apply(lambda x: clean_text(x))
    # 3. Create a column with no repetition disfluency
    data['transcript_without_repetition'] = data.transcript_with_tags.apply(lambda x: remove_repetition(x))
    data['transcript_without_repetition'] = data.transcript_without_repetition.apply(lambda x: remove_tags(x))
    data['transcript_without_repetition'] = data.transcript_without_repetition.apply(lambda x: clean_text(x))
    # 4. Create a column with no retracing disfluency
    data['transcript_without_retracing'] = data.transcript_with_tags.apply(lambda x: remove_retracing(x))
    data['transcript_without_retracing'] = data.transcript_without_retracing.apply(lambda x: remove_tags(x))
    data['transcript_without_retracing'] = data.transcript_without_retracing.apply(lambda x: clean_text(x))
    # 5. Create a column with no repetition/retracing disfluency
    data['transcript_without_either'] = data.transcript_with_tags.apply(lambda x: remove_repetition(x))
    data['transcript_without_either'] = data.transcript_without_either.apply(lambda x: remove_retracing(x))
    data['transcript_without_either'] = data.transcript_without_either.apply(lambda x: remove_tags(x))
    data['transcript_without_either'] = data.transcript_without_either.apply(lambda x: clean_text(x))
    # 6. Save Statistics
    # Counts
    data['total_words'] = data.transcript_without_tags.apply(lambda x: count_words(x))
    data['num_words_rep'] = data.transcript_without_repetition.apply(lambda x: count_words(x))
    data['num_words_ret'] = data.transcript_without_retracing.apply(lambda x: count_words(x))
    # Percentage
    data['rep_del'] = data.apply(lambda x: get_percentage(x.num_words_rep, x.total_words), axis=1)
    data['ret_del'] = data.apply(lambda x: get_percentage(x.num_words_ret, x.total_words), axis=1)

    # Average
    total_words = sum(data.total_words.tolist())
    num_words_rep = sum(data.num_words_rep.tolist())
    num_words_ret = sum(data.num_words_ret.tolist())
    num_words_both = sum(data.num_words_both.tolist())

    print("Rep: ", (total_words - num_words_rep) / total_words)
    print("Ret: ", (total_words - num_words_ret) / total_words)
    print("Both: ", (total_words - num_words_both) / total_words)

    rep_del = [val for val in data.rep_del.tolist() if val != -1]
    ret_del = [val for val in data.ret_del.tolist() if val != -1]

    avg_rep_del = np.mean(rep_del)
    avg_ret_del = np.mean(ret_del)
    print("Rep: ", avg_rep_del)
    print("Ret: ", avg_ret_del)

    # Speaker Level Data
    data_final = merge_speaker_utterances(data)

    data_final['num_words_both'] = data_final.transcript_without_disfluency.apply(lambda x: count_words(x))

    data_final['both_del'] = data_final.apply(lambda x: get_percentage(x.num_words_both, x.total_words), axis=1)

    # Speaker Level Statistics
    both_del = [val for val in data_final.both_del.tolist() if val != -1]
    avg_both_del = np.mean(both_del)
    print("Both: ", avg_both_del)

    # Save Data to File
    data[['speaker', 'transcript_without_tags', 'transcript_without_disfluency', 'dx', 'mmse']].to_pickle(
        "data2020fisher_utt_all.pickle")
    data_final.to_pickle('data2020fisher_all.pickle')


if __name__ == '__main__':
    main()
