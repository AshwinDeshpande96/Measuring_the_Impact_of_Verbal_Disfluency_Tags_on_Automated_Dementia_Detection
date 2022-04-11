import re


def remove_retracing(transcript):
    transcript_tokens = re.split("\s", transcript)
    phrase_retrace_errors = []
    word_retrace_errors = []
    for idx, token in enumerate(transcript_tokens):
        if token == "[//]":
            if idx < 2:
                continue
            if ">" in transcript_tokens[idx - 1]:
                error_i = [idx - 1]
                back_idx = idx - 1
                while back_idx > 0:
                    t = transcript_tokens[back_idx]
                    if back_idx not in error_i:
                        error_i = [back_idx] + error_i
                    if "<" in t:
                        break
                    back_idx -= 1
                phrase_retrace_errors.append(error_i)
            else:
                error_i = [idx - 1]
                word_retrace_errors.append(error_i)
                phrase_retrace_errors.append(error_i)
    for wi_list in word_retrace_errors + phrase_retrace_errors:
        for wi in wi_list:
            transcript_tokens[wi] = '[TO BE REMOVED]'
    transcript_tokens = [w for w in transcript_tokens if w != "[TO BE REMOVED]"]
    transcript_without_retrace = " ".join(transcript_tokens)
    return transcript_without_retrace
