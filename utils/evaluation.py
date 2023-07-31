import pandas as pd

def check_similarity(preds_row, targets_row):
    result = []
    strict_accum = 0
    strict_prec = 0
    flex_accum = 0
    flex_prec = 0

    for pred in preds_row:
        pred = str(pred).lower().strip()
        state = 'no_match'

        for target in targets_row:
            target = str(target).lower().strip()

            if pred == target:
                state = 'full_match'
                flex_accum += 1
                strict_accum += 1
                break
            elif pred in target:
                state = 'partial_match'
                flex_accum += 1
                strict_accum += 0.5
                break
            # perlu ditambahkan jika pred ada terdiri dari bbrp kata, dan katanya berisisan dengan target, maka menjadi partial match
            '''
            elif target in pred:
                state = 'partial_match'
                flex_accum += 1
                strict_accum += 0.5
            '''
        result.append(state)

    len_targets_row = len(targets_row)
    len_preds_row = len(preds_row)

    if len_preds_row != 0:
        flex_recall = flex_accum / len_targets_row
        flex_prec = flex_accum / len_preds_row

    if len_targets_row != 0:
        strict_recall = strict_accum / len_targets_row
        strict_prec = strict_accum / len_preds_row

    result.extend([strict_recall, strict_prec, flex_recall, flex_prec])

    return result

def eval(predictions, targets, to_dataframe = False):
    result = []

    for i in range(len(predictions)):
      result.append(check_similarity(predictions[i], targets[i]))
    
    if to_dataframe:
        return pd.DataFrame(result)
    else:
        return result
    
def evaluate_prediction(predict_list, targets_list):
    '''
    fungsi untuk membandingkan hasil prediksi dengan gold truth
    dengan hasil exact match =1, partial match = 1, no match = 0, utk kemudian dihitung per row
    '''
    evaluation = eval(predict_list, targets_list, True).round(3)
    evaluation.columns = ['key_1', 'key_2','key_3','strict_recall', 'strict_prec', 'flex_recall','flex_prec']
    evaluation = evaluation[['key_1', 'key_2','key_3', 'flex_recall','flex_prec']]
    return evaluation

def summarize_evaluation(eval_method):
    '''
    fungsi untuk menghitung score metric recall, precision, dan F1
    dengan paramater flexible : exact match =1, partial match = 1, no match = 0
    hasil dalam bentuk dataframe summary
    '''
    recall = eval_method['flex_recall'].mean()
    prec = eval_method['flex_prec'].mean()
    f1 = 2 * (prec * recall) / (prec + recall)

    summary = pd.DataFrame({'metric': [recall, prec, f1]}, index=['recall', 'precision', 'F1'])
    summary = summary.round(3)
    return summary

'''
blm di uji
----------------
'''

def evaluate_pred_vs_goldtruth(df_preds, df_targets):
    '''
    fungsi untuk membuat evaluasi hasil prediksi dengan gold truth
    evaluasi disajikan dalam 2 bentuk dataframe
    1. summary : recall/precision/F1
    2. detail : keyword prediksi, score, groundtruth, dan hasil mathc/partial match/no match
    '''
    # Evaluation w2v_TextRank Unigram
    df_preds_list = df_preds[['key_1','key_2','key_3']].values.tolist()
    df_targets_list = df_targets[["k1", "k2", "k3","k4", "k5", "k6","k7"]].values.tolist()
    eval_preds = eval(df_preds_list, df_targets_list, True).round(3)
    eval_preds.columns = ['key_1', 'key_2','key_3','strict_recall', 'strict_prec', 'flex_recall','flex_prec']
    eval_preds = eval_preds[['key_1', 'key_2','key_3', 'flex_recall','flex_prec']] # untuk menyederhanakan hasil evaluasi

    # Calculate w2v_TextRank Score, using flexible score : exact maatch =1, partial match = 1, no match = 0
    pred_recall = eval_preds['flex_recall'].mean()
    pred_prec = eval_preds['flex_prec'].mean()
    pred_f1 = 2 * (pred_prec * pred_recall) / (pred_prec + pred_recall)

    # Create a DataFrame with the scores
    summary = pd.DataFrame({'evaluation_preds': [pred_recall, pred_prec, pred_f1]}, \
                           index=['recall', 'precision', 'F1'])
    summary = summary.round(3)

    # Combine dataframe predict_w2v_textrank, df_targets and predict_w2v_textrank
    df_predict_evals = pd.concat([df_preds, df_targets, eval_preds], axis=1)
    
    return summary, df_predict_evals, eval_preds


