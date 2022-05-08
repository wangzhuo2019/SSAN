from sklearn.metrics import roc_curve, auc
import numpy as np


def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    
    return err, best_th, right_index

def performances_val(map_score_val_filename):
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for line in lines:
        try:
            count += 1
            tokens = line.split()
            score = float(tokens[0])
            label = float(tokens[1])  # int(tokens[1])
            val_scores.append(score)
            val_labels.append(label)
            data.append({'map_score': score, 'label': label})
            if label==1:
                num_real += 1
            else:
                num_fake += 1
        except:
            continue
    
    fpr,tpr,threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_test = auc(fpr, tpr)
    val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)
    
    type1 = len([s for s in data if s['map_score'] < val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])
    
    val_ACC = 1-(type1 + type2) / count
    
    FRR = 1- tpr    # FRR = 1 - TPR
    
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate
    
    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], auc_test, val_err


def performances_tpr_fpr(map_score_val_filename):
    with open(map_score_val_filename, 'r') as file:
        lines = file.readlines()
    scores = []
    labels = []
    for line in lines:
        try:
            record = line.split()
            scores.append(float(record[0]))
            labels.append(float(record[1]))
        except:
            continue

    fpr_list = [0.1, 0.01, 0.001, 0.0001]
    threshold_list = get_thresholdtable_from_fpr(scores,labels, fpr_list)
    tpr_list = get_tpr_from_threshold(scores,labels, threshold_list)
    return tpr_list


def get_thresholdtable_from_fpr(scores, labels, fpr_list):
    threshold_list = []
    live_scores = []
    for score, label in zip(scores,labels):
        if label == 1:
            live_scores.append(float(score))
    live_scores.sort()
    live_nums = len(live_scores)
    for fpr in fpr_list:
        i_sample = int(fpr * live_nums)
        i_sample = max(1, i_sample)
        if not live_scores:
            return [0.5]*10
        threshold_list.append(live_scores[i_sample - 1])
    return threshold_list

# Get the threshold under thresholds
def get_tpr_from_threshold(scores,labels, threshold_list):
    tpr_list = []
    hack_scores = []
    for score, label in zip(scores,labels):
        if label == 0:
            hack_scores.append(float(score))
    hack_scores.sort()
    hack_nums = len(hack_scores)
    for threshold in threshold_list:
        hack_index = 0
        while hack_index < hack_nums:
            if hack_scores[hack_index] >= threshold:
                break
            else:
                hack_index += 1
        if hack_nums != 0:
            tpr = hack_index * 1.0 / hack_nums
        else:
            tpr = 0
        tpr_list.append(tpr)
    return tpr_list