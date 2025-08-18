import numpy as np
from sklearn.metrics import average_precision_score


def get_map(y_true, y_pred_probs_list):
    average_precisions = []

    true_labels = np.concatenate([np.array(x) for x in y_true])
    predicted_probabilities = np.concatenate([np.array(x) for x in y_pred_probs_list])
    for class_idx in range(true_labels.shape[1]):
        class_true = true_labels[:, class_idx]
        class_scores = predicted_probabilities[:, class_idx]
        average_precision = average_precision_score(class_true, class_scores)
        average_precisions.append(average_precision)

    # Calculate the mean of the average precisions across all classes to obtain mAP
    mAP = np.mean(average_precisions)
    C1_ap = average_precisions[0]
    C2_ap = average_precisions[1]
    C3_ap = average_precisions[2]
    
    return C1_ap, C2_ap, C3_ap, mAP