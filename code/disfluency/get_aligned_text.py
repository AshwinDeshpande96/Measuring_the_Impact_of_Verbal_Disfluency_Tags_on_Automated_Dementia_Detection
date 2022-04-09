import re

import h5py
import numpy as np
import pandas as pd

columns = ['reference_text', 'asr_text']
df = pd.DataFrame(columns=columns)
new_data = pd.read_pickle('new_data3.pickle')
with open("results2.txt", "r") as fptr:
    lines = fptr.read().split("\n")
alignment = {}
i = 0
sent_num = 1
rtoken_to_name = {"[/]": "REP",
                  "[//]": "RET"}
while sent_num <= 1492:
    line = lines[i]
    if not line.strip() == str(sent_num):
        i += 1
        continue
    ref = lines[i + 1].replace("REF: \t", "").split()
    asr = lines[i + 2].replace("HYP: \t", "").split()
    row = {"reference_text": ref,
           "asr_text": asr}
    # print(f"############{sent_num}\n\t{ref}\n\t{asr}")
    df = df.append(row, ignore_index=True)
    alignment[sent_num] = (ref, asr)
    sent_num += 1
    i += 3
data = pd.concat([new_data, df], axis=1)
print()


def clean_text(text):
    text = re.sub(r"\[.*\]", "", text)
    text = re.sub(r"[^a-zA-Z' ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_dict(keys, values):
    thedict = {}
    for k, v in zip(keys, values):
        i = 1
        key = f"{k}_{i}"
        while key in thedict:
            i += 1
            key = f"{k}_{i}"
        thedict[key] = v
    return thedict


def make_unique(asr_text):
    ulist = []
    for k in asr_text:
        i = 1
        key = f"{k}_{i}"
        while key in ulist:
            i += 1
            key = f"{k}_{i}"
        ulist.append(key)
    return ulist


def align_words(utext, uorig_text):
    starter = -1
    matched_indices = []
    for token in utext:
        try:
            sub_original_tokens = [None for _ in range(starter + 1)] + uorig_text[starter + 1:]
            token_index = sub_original_tokens.index(token)
            matched_indices.append(token_index)
            starter = token_index
        except:
            matched_indices.append(None)
    alignments = dict(zip(utext, matched_indices))
    return alignments


def get_checklist(both_errors):
    checklist_dict = {}
    for rtoken, index_set in both_errors.items():
        checklist_set = []
        for indices in index_set:
            checklist = dict(zip(indices, [False for _ in range(len(indices))]))
            error_indices = dict(zip(indices, [0 for _ in range(len(indices))]))
            checklist_set.append({"checklist": checklist,
                                  "error_indices": error_indices})
        checklist_dict[rtoken_to_name[rtoken]] = checklist_set
    return checklist_dict


def get_map(x, columns):
    disfluency = x.disfluency_asr
    reference_text = x.reference_text
    both_reference_text = x.both_reference_text
    asr_text = x.asr_text
    etype = x.both_etype
    error_checklist = get_checklist(x.both_errors)

    if not isinstance(disfluency, str):
        return pd.Series([None, error_checklist], index=columns)
    if not isinstance(asr_text, list):
        return pd.Series([None, error_checklist], index=columns)
    words = disfluency.split()[::2]
    tags = disfluency.split()[1::2]
    word_to_tag = make_dict(words, tags)
    uasr_text = make_unique(asr_text)
    uref_text = make_unique(reference_text)
    uorig_text = make_unique(both_reference_text)
    asr_alignment = align_words(uasr_text, uorig_text)
    ref_alignment = align_words(uref_text, uorig_text)
    result_map = {}
    for idx, asr_word, ref_word, uasr_word, uref_word in zip(list(range(len(asr_text))),
                                                             asr_text,
                                                             reference_text,
                                                             uasr_text,
                                                             uref_text):
        ref_word_idx = ref_alignment.get(uref_word, None)
        fisher_tag = word_to_tag.get(uasr_word, None)
        e = etype.get(uref_word, None)
        result_map[idx] = [asr_word, fisher_tag, ref_word, e, "INCOMPLETE"]
        if e in error_checklist:
            for phrase_checklist in error_checklist[e]:
                if ref_word_idx in phrase_checklist['checklist'] and fisher_tag == "E":
                    if asr_word != ref_word.lower():
                        print("What?")
                    phrase_checklist['checklist'][ref_word_idx] = True
                    phrase_checklist['error_indices'][ref_word_idx] = idx
                    if sum(phrase_checklist['checklist'].values()) == len(phrase_checklist['checklist']):
                        phrase_indices = phrase_checklist['error_indices'].values()
                        for i in phrase_indices:
                            rs = result_map[i]
                            result_map[i] = rs[:-1] + ["COMPLETE"]
    return pd.Series([list(result_map.values()), error_checklist], index=columns)


data[['asr_text_map', 'error_checklist']] = data.apply(lambda x: get_map(x, ['asr_text_map', 'error_checklist']),
                                                       axis=1)


def remove_disfluency(x, errors):
    asr_text_map = x.asr_text_map
    # asr_text_map = literal_eval(asr_text_map)
    if not isinstance(asr_text_map, list):
        return None
    tokens = []
    for asr_word, tag, ref_text, etype, complete_status in asr_text_map:
        if tag == "E" and re.match(r"^[A-Z]+$", ref_text) and complete_status == "COMPLETE" and etype in errors:
            continue
        elif re.match(r"^\*+$", asr_word):
            continue
        tokens.append(asr_word)
    tokens = " ".join(tokens)
    return tokens


def get_asr_text(x):
    disfluency_tags = x.disfluency_asr
    if disfluency_tags is None or not disfluency_tags:
        return None
    tokens = disfluency_tags.split()
    words = tokens[::2]
    text = " ".join(words)
    return text


def get_etagged(x):
    disfluency_tags = x.disfluency_asr
    if disfluency_tags is None or not disfluency_tags:
        return None
    disfluency_tags = re.sub(r"\_", "", disfluency_tags)
    disfluency_tags = re.sub(r"\s+", " ", disfluency_tags).strip()
    return disfluency_tags


data['all_errors'] = data.apply(lambda x: get_asr_text(x), axis=1)
data['e_tagged'] = data.apply(lambda x: get_etagged(x), axis=1)
data['rep_eremoved'] = data.apply(lambda x: remove_disfluency(x, ["REP"]), axis=1)
data['ret_eremoved'] = data.apply(lambda x: remove_disfluency(x, ["RET"]), axis=1)
data['both_eremoved'] = data.apply(lambda x: remove_disfluency(x, ["REP", "RET"]), axis=1)


def get_counts(x, columns):
    errors = x.error_checklist
    phrase_rep_count_asr = 0
    word_rep_count_asr = 0
    phrase_ret_count_asr = 0
    word_ret_count_asr = 0

    phrase_rep_count_gold = 0
    word_rep_count_gold = 0
    phrase_ret_count_gold = 0
    word_ret_count_gold = 0
    for check in errors['REP']:
        if len(check['checklist']) == 1:
            if sum(check['checklist'].values()) == len(check['checklist']):
                word_rep_count_asr += 1
            word_rep_count_gold += 1
        elif len(check['checklist']) > 1:
            if sum(check['checklist'].values()) == len(check['checklist']):
                phrase_rep_count_asr += 1
            phrase_rep_count_gold += 1
    for check in errors['RET']:
        if len(check['checklist']) == 1:
            if sum(check['checklist'].values()) == len(check['checklist']):
                word_ret_count_asr += 1
            word_ret_count_gold += 1
        elif len(check['checklist']) > 1:
            if sum(check['checklist'].values()) == len(check['checklist']):
                phrase_ret_count_asr += 1
            phrase_ret_count_gold += 1
    return pd.Series([phrase_rep_count_asr,
                      word_rep_count_asr,
                      phrase_ret_count_asr,
                      word_ret_count_asr,
                      phrase_rep_count_gold,
                      word_rep_count_gold,
                      phrase_ret_count_gold,
                      word_ret_count_gold], index=columns)


count_columns = ['phrase_rep_count_asr',
                 'word_rep_count_asr',
                 'phrase_ret_count_asr',
                 'word_ret_count_asr',
                 'phrase_rep_count_gold',
                 'word_rep_count_gold',
                 'phrase_ret_count_gold',
                 'word_ret_count_gold'
                 ]
data[count_columns] = data.apply(lambda x: get_counts(x, count_columns),
                                 axis=1)

# data_ad = data.loc[data['dx'] == "Dementia"][count_columns]
# data_cn = data.loc[data['dx'] == "Control"][count_columns]
# ad_phrase_rep_count = sum(data_ad['phrase_rep_count'].tolist())
# ad_word_rep_count = sum(data_ad['word_rep_count'].tolist())
# ad_phrase_ret_count = sum(data_ad['phrase_ret_count'].tolist())
# ad_word_ret_count = sum(data_ad['word_ret_count'].tolist())
#
# cn_phrase_rep_count = sum(data_cn['phrase_rep_count'].tolist())
# cn_word_rep_count = sum(data_cn['word_rep_count'].tolist())
# cn_phrase_ret_count = sum(data_cn['phrase_ret_count'].tolist())
# cn_word_ret_count = sum(data_cn['word_ret_count'].tolist())
# print("ad_phrase_rep_count", ad_phrase_rep_count)
# print("ad_word_rep_count", ad_word_rep_count)
# print("ad_phrase_ret_count", ad_phrase_ret_count)
# print("ad_word_ret_count", ad_word_ret_count)
# print("cn_phrase_rep_count", cn_phrase_rep_count)
# print("cn_word_rep_count", cn_word_rep_count)
# print("cn_phrase_ret_count", cn_phrase_ret_count)
# print("cn_word_ret_count", cn_word_ret_count)
#
# avg_rep_count = np.mean(data['phrase_rep_count'].tolist() + data['word_rep_count'].tolist())
# avg_ret_count = np.mean(data['phrase_ret_count'].tolist() + data['word_ret_count'].tolist())
# print(avg_rep_count, avg_ret_count)
data_final = {}
string_columns = ['all_errors',
                  'e_tagged',
                  'rep_eremoved',
                  'ret_eremoved',
                  'both_eremoved']
number_columns = count_columns
new_columns = ['speaker'] + string_columns + number_columns + ['dx', 'mmse']
string_idx = len(string_columns) + 1
num_idx = string_idx + len(number_columns)
for (idx, (row)) in data[new_columns].iterrows():
    row = row.tolist()
    stringrow = ["" if not isinstance(r, str) or r == "placeholder" else r for r in row[1:string_idx]]
    numrow = [0 if not type(num) in [int, float] else num for num in row[string_idx:num_idx]]
    row = row[0:1] + stringrow + numrow + row[num_idx:]
    speaker = row[0]
    if speaker not in data_final:
        data_final[speaker] = row
    else:
        old_rows_string = data_final[speaker][1:string_idx]
        old_rows_num = data_final[speaker][string_idx:num_idx]
        new_rows_string = [(old + ". " + new) if new else old for new, old in
                           zip(row[1:string_idx], old_rows_string)]
        new_rows_num = [(old + new) for new, old in zip(row[string_idx:num_idx], old_rows_num)]
        new_rows = row[0:1] + new_rows_string + new_rows_num + row[num_idx:]
        data_final[speaker] = new_rows

data_final = pd.DataFrame(list(data_final.values()), columns=new_columns)

data_ad = data_final.loc[data_final['dx'] == "Dementia"][count_columns]
data_cn = data_final.loc[data_final['dx'] == "Control"][count_columns]
ad_phrase_rep_count = sum(data_ad['phrase_rep_count_asr'].tolist())
ad_word_rep_count = sum(data_ad['word_rep_count_asr'].tolist())
ad_phrase_ret_count = sum(data_ad['phrase_ret_count_asr'].tolist())
ad_word_ret_count = sum(data_ad['word_ret_count_asr'].tolist())

cn_phrase_rep_count = sum(data_cn['phrase_rep_count_asr'].tolist())
cn_word_rep_count = sum(data_cn['word_rep_count_asr'].tolist())
cn_phrase_ret_count = sum(data_cn['phrase_ret_count_asr'].tolist())
cn_word_ret_count = sum(data_cn['word_ret_count_asr'].tolist())
print("######### asr")
print("ad_word_rep_count", ad_word_rep_count)
print("ad_phrase_rep_count", ad_phrase_rep_count)
print("ad_word_ret_count", ad_word_ret_count)
print("ad_phrase_ret_count", ad_phrase_ret_count)
print()
print("cn_word_rep_count", cn_word_rep_count)
print("cn_phrase_rep_count", cn_phrase_rep_count)
print("cn_word_ret_count", cn_word_ret_count)
print("cn_phrase_ret_count", cn_phrase_ret_count)
data_final['rep_count_asr'] = data_final.apply(lambda x: x.phrase_rep_count_asr + x.word_rep_count_asr, axis=1)
data_final['ret_count_asr'] = data_final.apply(lambda x: x.phrase_ret_count_asr + x.word_ret_count_asr, axis=1)

ad_phrase_rep_count = sum(data_ad['phrase_rep_count_gold'].tolist())
ad_word_rep_count = sum(data_ad['word_rep_count_gold'].tolist())
ad_phrase_ret_count = sum(data_ad['phrase_ret_count_gold'].tolist())
ad_word_ret_count = sum(data_ad['word_ret_count_gold'].tolist())

cn_phrase_rep_count = sum(data_cn['phrase_rep_count_gold'].tolist())
cn_word_rep_count = sum(data_cn['word_rep_count_gold'].tolist())
cn_phrase_ret_count = sum(data_cn['phrase_ret_count_gold'].tolist())
cn_word_ret_count = sum(data_cn['word_ret_count_gold'].tolist())
print("######### gold")
print("ad_word_rep_count", ad_word_rep_count)
print("ad_phrase_rep_count", ad_phrase_rep_count)
print("ad_word_ret_count", ad_word_ret_count)
print("ad_phrase_ret_count", ad_phrase_ret_count)
print()
print("cn_word_rep_count", cn_word_rep_count)
print("cn_phrase_rep_count", cn_phrase_rep_count)
print("cn_word_ret_count", cn_word_ret_count)
print("cn_phrase_ret_count", cn_phrase_ret_count)
data_final['rep_count_gold'] = data_final.apply(lambda x: x.phrase_rep_count_gold + x.word_rep_count_gold, axis=1)
data_final['ret_count_gold'] = data_final.apply(lambda x: x.phrase_ret_count_gold + x.word_ret_count_gold, axis=1)

print("Repetition gold: ", np.mean(data_final['rep_count_gold'].tolist()))
print("Retracing gold: ", np.mean(data_final['ret_count_gold'].tolist()))
print("Repetition asr: ", np.mean(data_final['rep_count_asr'].tolist()))
print("Retracing asr: ", np.mean(data_final['ret_count_asr'].tolist()))


def save_to_h5(filename, data):
    hf = h5py.File(filename, 'w')
    hf.create_dataset('default', data=data)


data_final.to_csv("counts.csv")


# data_final.to_pickle("counts.pickle")


def get_percent(asr_count, gold_count):
    if gold_count == 0:
        return None
    return asr_count / gold_count


data_final['word_rep_percent'] = data_final.apply(lambda x: get_percent(x.word_rep_count_asr,
                                                                        x.word_rep_count_gold),
                                                  axis=1)
data_final['phrase_rep_percent'] = data_final.apply(lambda x: get_percent(x.phrase_rep_count_asr,
                                                                          x.phrase_rep_count_gold),
                                                    axis=1)
data_final['word_ret_percent'] = data_final.apply(lambda x: get_percent(x.word_ret_count_asr,
                                                                        x.word_ret_count_gold),
                                                  axis=1)
data_final['phrase_ret_percent'] = data_final.apply(lambda x: get_percent(x.phrase_ret_count_asr,
                                                                          x.phrase_ret_count_gold),
                                                    axis=1)

print("Word Rep %: ", np.mean(data_final.word_rep_percent.dropna().tolist()))
print("Phrase Rep %: ", np.mean(data_final.phrase_rep_percent.dropna().tolist()))
print("Word Ret %: ", np.mean(data_final.word_ret_percent.dropna().tolist()))
print("Phrase Ret %: ", np.mean(data_final.phrase_ret_percent.dropna().tolist()))
# data_final['avg_rep'] = data_final.apply(lambda x: divide(x.rep_count, x.total_count), axis=1)
# data_final['avg_ret'] = data_final.apply(lambda x: divide(x.ret_count, x.total_count), axis=1)
#
# avg_rep_count = np.mean(data_final['avg_rep'].tolist())
# avg_ret_count = np.mean(data_final['avg_ret'].tolist())
# print(avg_rep_count, avg_ret_count)
# # data_final.to_pickle("new_data.pickle")
#
#
# def count_words(text):
#     if text is None:
#         return
#     text = text.split()
#     return len(text)
#
#
# def count_disfluency(text, total):
#     if text is None:
#         return -1
#     text = text.split()
#     return total - len(text)

# data_final['total_words'] = data_final.all_errors.apply(lambda x: count_words(x))
# data_final['num_words_rep'] = data_final.apply(
#     lambda x: count_disfluency(x.rep_eremoved, x.total_words), axis=1)
# data_final['num_words_ret'] = data_final.apply(
#     lambda x: count_disfluency(x.ret_eremoved, x.total_words), axis=1)
#
# rep_del = [val for val in data_final.num_words_rep.tolist() if val != -1]
# ret_del = [val for val in data_final.num_words_ret.tolist() if val != -1]
#
# avg_rep_del = np.mean(rep_del)
# avg_ret_del = np.mean(ret_del)
#
# print(avg_rep_del)
# print(avg_ret_del)
# print()
