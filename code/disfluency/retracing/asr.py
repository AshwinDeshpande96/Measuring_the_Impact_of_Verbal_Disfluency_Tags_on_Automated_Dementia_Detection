from ..util import update


def remove_retracing(pred, alignment, all_errors):
    if all_errors is None or not all_errors:
        if pred is not None:
            pred = pred.split()
            transcript = " ".join([w for w in pred[::2]])
            transcript = ""
            return transcript
        else:
            return None
    if pred is None or not pred:
        return None
    pred = pred.split()
    pred_tags = pred[1::2]
    pred_words = pred[::2]
    detected_wret_errors = {}
    detected_wret_idx = {}
    detected_pret_errors = {}
    detected_pret_idx = {}
    _, _, target_wrets, target_prets = all_errors
    for (idx, (word, orig_idx), tag) in zip(list(range(len(alignment))), alignment, pred_tags):
        if tag != "E":
            continue
        for wrep in target_wrets:
            if orig_idx in wrep:
                key = "_".join([str(wi) for wi in wrep])
                val = [True if wi == orig_idx else False for wi in wrep]
                val2 = [idx if wi == orig_idx else None for wi in wrep]
                if key not in detected_wret_errors:
                    detected_wret_errors[key] = val
                    detected_wret_idx[key] = val2
                else:
                    value = detected_wret_errors[key]
                    value2 = detected_wret_idx[key]
                    value = [new or old for new, old in zip(val, value)]
                    value2 = [update(new, old) for new, old in zip(val2, value2)]
                    detected_wret_errors[key] = value
                    detected_wret_idx[key] = value2
        for prep in target_prets:
            if orig_idx in prep:
                key = "_".join([str(wi) for wi in prep])
                val = [True if wi == orig_idx else False for wi in prep]
                val2 = [idx if wi == orig_idx else None for wi in prep]
                if key not in detected_pret_errors:
                    detected_pret_errors[key] = val
                    detected_pret_idx[key] = val2
                else:
                    value = detected_pret_errors[key]
                    value2 = detected_pret_idx[key]
                    value = [new or old for new, old in zip(val, value)]
                    value2 = [update(new, old) for new, old in zip(val2, value2)]
                    detected_pret_errors[key] = value
                    detected_pret_idx[key] = value2
    success_wrep = []
    success_prep = []
    for key, value in detected_wret_errors.items():
        if sum(value) == len(value):
            success_wrep.append(key)
    for key, value in detected_pret_errors.items():
        if sum(value) == len(value):
            success_prep.append(key)
    for key in success_wrep:
        words_to_remove = detected_wret_idx[key]
        for wi in words_to_remove:
            pred_words[wi] = "[TO BE REMOVED]"
    for key in success_prep:
        words_to_remove = detected_pret_idx[key]
        for wi in words_to_remove:
            pred_words[wi] = "[TO BE REMOVED]"
    transcript = " ".join([w for w in pred_words if w != '[TO BE REMOVED]'])
    return transcript
