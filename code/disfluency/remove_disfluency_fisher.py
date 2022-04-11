import re
from itertools import groupby

import numpy as np
import pandas as pd

from repetition.asr import remove_repetition
from retracing.asr import remove_retracing
from util import count_words, get_percentage


def get_words(text):
    if text is None:
        return
    tokens = text.split()
    twords = tokens[::2]
    transcript = " ".join(twords)
    return transcript


def remove_underscore(text):
    if text is None:
        return
    text = re.sub(r"\_", "F", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_duplicates(text):
    text = [i[0] for i in groupby(text.split())]
    text = " ".join(text)
    return text


def remove_disfluency(text):
    if text is None:
        return
    text = text.split()
    text_tags = text[1::2]
    text_words = text[::2]
    text = ""
    for w, t in zip(text_words, text_tags):
        if t == "E":
            continue
        text += " " + w
    return text


def remove_filled_pauses(text):
    if text is None:
        return
    text = re.sub(r"\b[uhmo]{2,}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = remove_duplicates(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def merge_speaker_utterances(data):
    data_final = {}
    for (idx, (speaker,
               transcript_without_tags,
               transcript_without_disfluency,
               transcript_without_underscore,
               transcript_without_repetition,
               transcript_without_retracing,
               dx,
               mmse)) in data[['speaker',
                               'transcript_without_tags',
                               'transcript_without_disfluency',
                               'transcript_without_underscore',
                               'transcript_without_repetition',
                               'transcript_without_retracing',
                               'dx',
                               'mmse']].iterrows():
        if speaker not in data_final:
            data_final[speaker] = (speaker,
                                   transcript_without_tags,
                                   transcript_without_disfluency,
                                   transcript_without_underscore,
                                   transcript_without_repetition,
                                   transcript_without_retracing,
                                   dx,
                                   mmse)
        else:
            _transcript_without_tags = data_final[speaker][1]
            _transcript_without_disfluency = data_final[speaker][2]
            _transcript_without_underscore = data_final[speaker][3]
            _transcript_without_repetition = data_final[speaker][4]
            _transcript_without_retracing = data_final[speaker][5]
            if transcript_without_tags is not None:
                _transcript_without_tags += " " + transcript_without_tags
            if transcript_without_disfluency is not None:
                _transcript_without_disfluency += " " + transcript_without_disfluency
            if transcript_without_underscore is not None:
                _transcript_without_underscore += " " + transcript_without_underscore
            if transcript_without_repetition is not None:
                _transcript_without_repetition += " " + transcript_without_repetition
            if transcript_without_retracing is not None:
                _transcript_without_retracing += " " + transcript_without_retracing
            data_final[speaker] = (speaker,
                                   _transcript_without_tags,
                                   _transcript_without_disfluency,
                                   _transcript_without_underscore,
                                   _transcript_without_repetition,
                                   _transcript_without_retracing,
                                   dx,
                                   mmse)

    data_final = pd.DataFrame(list(data_final.values()), columns=['speaker',
                                                                  'transcript_without_tags',
                                                                  'transcript_without_disfluency',
                                                                  'transcript_without_underscore',
                                                                  'transcript_without_repetition',
                                                                  'transcript_without_retracing',
                                                                  'dx',
                                                                  'mmse'])
    return data_final


def main():
    # Import Data
    data = pd.read_pickle('../transcript_with_disfluency_parse_train.pickle')

    # Initial Summary
    classwise_Etags = {"Control": 0, "Dementia": 0}
    classwise_percentage = {"Control": 0, "Dementia": 0}
    classwise_count = {"Control": 0, "Dementia": 0}
    for (idx, (disfluency_text, dx)) in data[['disfluency_text', 'dx']].iterrows():
        if disfluency_text is None:
            continue
        tokens = disfluency_text.split()
        tags = tokens[1::2]
        e_tokens = [t for t in tags if t == "E"]
        classwise_Etags[dx] += len(e_tokens)
        classwise_percentage[dx] += len(e_tokens) / len(tags)
        classwise_count[dx] += 1
    print(classwise_Etags)
    print("Control AVG%: ", classwise_percentage["Control"] / classwise_count["Control"])
    print("Dementia AVG%: ", classwise_percentage["Dementia"] / classwise_count["Dementia"])

    # 1. Add MMSE score to each utterance
    df = pd.read_csv('data.csv', usecols=['ID', 'mmse'])
    id_to_mmse = df.set_index('ID')['mmse'].to_dict()
    data['mmse'] = data.speaker.apply(lambda x: id_to_mmse.get(x, None))

    # 2. Create a column with no disfluency markers
    data['transcript_without_tags'] = data.disfluency_text.apply(lambda x: get_words(x))

    # 3. Create a column with disfluency markers (Fluent: F, Disfluent: E)
    data['transcript_without_underscore'] = data.disfluency_text.apply(lambda x: remove_underscore(x))

    # 4. Create a column with no repetition disfluency
    data['transcript_without_repetition'] = data.apply(lambda x: remove_repetition(x.disfluency_text,
                                                                                   x.alignments,
                                                                                   x.all_errors), axis=1)

    # 5. Create a column with no retracing disfluency
    data['transcript_without_retracing'] = data.apply(lambda x: remove_retracing(x.disfluency_text,
                                                                                 x.alignments,
                                                                                 x.all_errors), axis=1)

    # 6. Create a column with repetition/retracing disfluency
    data['transcript_without_disfluency'] = data.disfluency_text.apply(lambda x: remove_disfluency(x))

    # 7. Create a column with no filled pauses
    data['transcript_without_uh'] = data.transcript_without_disfluency.apply(lambda x: remove_filled_pauses(x))

    # 8. Save Utterance Level Data
    data[['speaker',
          'transcript_without_tags',
          'transcript_without_disfluency',
          'dx',
          'mmse']].to_pickle("data2020fisher_utt_all.pickle")

    # 9. Speaker Level Data
    data_final = merge_speaker_utterances(data)
    data_final.to_pickle('data2020fisher_all.pickle')
    data_final.to_csv('data2020fisher_all.csv')
    data[['speaker',
          'transcript_without_tags',
          'transcript_without_disfluency',
          'transcript_without_underscore',
          'transcript_without_repetition',
          'transcript_without_retracing',
          'dx',
          'mmse']].to_pickle("data2020fisher_utt_all.pickle")

    # 10. Speaker Level Statistics
    # Counts
    data_final['total_words'] = data_final.transcript_without_tags.apply(lambda x: count_words(x))
    data_final['num_words_rep'] = data_final.transcript_without_repetition.apply(lambda x: count_words(x))
    data_final['num_words_ret'] = data_final.transcript_without_retracing.apply(lambda x: count_words(x))
    data_final['num_words_both'] = data_final.transcript_without_disfluency.apply(lambda x: count_words(x))
    # Percentage
    data_final['rep_del'] = data_final.apply(lambda x: get_percentage(x.num_words_rep, x.total_words), axis=1)
    data_final['ret_del'] = data_final.apply(lambda x: get_percentage(x.num_words_ret, x.total_words), axis=1)
    data_final['both_del'] = data_final.apply(lambda x: get_percentage(x.num_words_both, x.total_words), axis=1)

    # Average
    total_words = sum(data.total_words.tolist())
    num_words_rep = sum(data.num_words_rep.tolist())
    num_words_ret = sum(data.num_words_ret.tolist())
    num_words_both = sum(data.num_words_both.tolist())

    print("Rep: ", (total_words - num_words_rep) / total_words)
    print("Ret: ", (total_words - num_words_ret) / total_words)
    print("Both: ", (total_words - num_words_both) / total_words)

    rep_del = [val for val in data_final.rep_del.tolist() if val != -1]
    ret_del = [val for val in data_final.ret_del.tolist() if val != -1]
    both_del = [val for val in data_final.both_del.tolist() if val != -1]
    avg_rep_del = np.mean(rep_del)
    avg_ret_del = np.mean(ret_del)
    avg_both_del = np.mean(both_del)

    print("Rep: ", avg_rep_del)
    print("Ret: ", avg_ret_del)
    print("Both: ", avg_both_del)

    # 11. Save Speaker Level Data
    data_final.to_pickle('data2020fisher_all.pickle')


if __name__ == '__main__':
    main()
