import re


def remove_repetition(transcript):
    if transcript == """*PAR:\t+" shh . \x1529046_29510\x15\n""":
        print()
    transcript_tokens = re.split("\s", transcript)
    phrase_repeat_errors = []
    word_repeat_errors = []
    for idx, token in enumerate(transcript_tokens):
        if token == "[/]":
            if idx < 2:
                continue
            if ">" in transcript_tokens[idx - 1]:
                # Phrase Repetition
                error_i = [idx - 1]
                back_idx = idx - 1
                while back_idx > 0:
                    t = transcript_tokens[back_idx]
                    if back_idx not in error_i:
                        error_i = [back_idx] + error_i
                    if "<" in t:
                        break
                    back_idx -= 1
                phrase_repeat_errors.append(error_i)
            else:
                # Word Repetition
                error_i = [idx - 1]
                word_repeat_errors.append(error_i)
                phrase_repeat_errors.append(error_i)
    for wi_list in word_repeat_errors + phrase_repeat_errors:
        for wi in wi_list:
            transcript_tokens[wi] = '[TO BE REMOVED]'
    transcript_tokens = [w for w in transcript_tokens if w != "[TO BE REMOVED]"]
    transcript_without_repetition = " ".join(transcript_tokens)
    transcript_without_repetition = re.sub(r"\[x[0-9 ]+\]", "", transcript_without_repetition)
    return transcript_without_repetition
