import numpy as np


party_dict = {
    'SPD': 0,
    'FDP': 1,
    'GRUENE': 2,
    'CDU/CSU': 3,
    'AfD': 4,
    'PDS/LINKE': 5,
    'independent': 6,
}


def accuracy(y_pred, y_true):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_preds = (y_pred_tags == y_true).float()
    acc = correct_preds.sum() / len(correct_preds)

    acc = torch.round(acc * 100)

    return acc


def remove_comments(speech, delimiters):
    paren = 0
    res = ''
    for ch in speech:
        if ch == delimiters[0]:
            paren += 1
        elif ch == delimiters[1] and paren == 1:
            paren -= 1
        elif not paren:
            res += ch

    return res


def cut_text(text_data, label_data, maxLength):
    ret_text = []
    ret_label = []

    for i, txt in enumerate(text_data):
        crrLbl = label_data[i]
        crrTxt = txt
        tmpTxt = []
        tmpLbl = []
        tmpTxt.append(crrTxt[:maxLength])
        tmpLbl.append(crrLbl)
        crrTxt = crrTxt[maxLength:]
        while len(crrTxt) > maxLength:
            tmpTxt.append(crrTxt[:maxLength])
            tmpLbl.append(crrLbl)
            crrTxt = crrTxt[maxLength:]

        ret_text.extend(tmpTxt)
        ret_label.extend(tmpLbl)

    return np.array(ret_text, dtype=object), np.array(ret_label)


def pad_text(text_data, seq_length):

    ret = np.zeros((len(text_data), seq_length), dtype=int)

    for i, txt in enumerate(text_data):
        lenTxt = len(txt)

        if lenTxt <= seq_length:
            ret[i, (seq_length - lenTxt):] = np.array(txt)
        else:
            ret[i, :] = np.array(txt[:seq_length])

    return ret
