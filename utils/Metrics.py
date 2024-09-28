import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score

def calculate_fom_for_tile(S_true_start, S_true_end, S_pred_end,unique_classes ,mask=None):   
    S_true_start = np.array(S_true_start)
    S_true_end = np.array(S_true_end)
    S_pred_end = np.array(S_pred_end) 

    mask = np.array(mask) 
    mask = mask.astype(bool)

    S_true_start_masked = S_true_start[:,mask]
    S_true_end_masked = S_true_end[:, mask]
    S_pred_end_masked = S_pred_end[:, mask]

    T, H, W = S_true_end.shape
    A = np.zeros(T+1)
    B = np.zeros(T+1)
    C = np.zeros(T+1)
    D = np.zeros(T+1)
    Fom=np.zeros(T+1)
    S_pred = [[] for _ in range(T+1)]
    S_true = [[] for _ in range(T+1)]
    total_pred=[]
    total_true=[]

    for t in range(T):  
        start_masked = S_true_start_masked
        true_end_t = S_true_end_masked[t]
        pred_end_t = S_pred_end_masked[t] 

        S_true[t+1]=true_end_t.flatten()     
        S_pred[t+1]=pred_end_t.flatten()
        total_true.extend(S_true[t])
        total_pred.extend(S_pred[t])   

        cur_Fom, cur_A, cur_B, cur_C, cur_D = calculate_fom(start_masked,true_end_t,pred_end_t)       
        A[t+1] = cur_A  
        B[t+1] = cur_B   
        C[t+1] = cur_C  
        D[t+1] = cur_D         
        Fom[t+1] = cur_Fom   
    A[0]=A.sum()
    B[0]=B.sum()
    C[0]=C.sum()
    D[0]=D.sum()
    sum_ABCD=A[0] +B[0] + C[0] + D[0]
    Fom[0]=B[0] / (sum_ABCD) if sum_ABCD != 0 else 0   

    S_pred[0]=total_pred
    S_true[0]=total_true
    accuracy_list=[]
    maxtric_list=[]   
    for x in range(T+1):
        cur_pred=S_pred[x]
        cur_true=S_true[x]
        overall_accuracy, confusion_df=calculate_metrics(cur_true,cur_pred,unique_classes)   
        accuracy_list.append(overall_accuracy)     
        maxtric_list.append(confusion_df)
    return Fom,A,B,C,D,accuracy_list,maxtric_list, S_true,S_pred


def calculate_fom(S_true_start, S_true_end, S_pred_end):   
    S_true_start = np.array(S_true_start)
    S_true_end = np.array(S_true_end)
    S_pred_end = np.array(S_pred_end)  

    A = np.sum((S_true_start != S_true_end) & (S_true_start == S_pred_end))  
    B = np.sum((S_true_start != S_true_end) & (S_true_end == S_pred_end))   
    C = np.sum((S_true_start != S_true_end) & (S_true_end != S_pred_end) & (S_true_start != S_pred_end))  
    D = np.sum((S_true_start == S_true_end) & (S_true_start != S_pred_end))  

    fom = B / (A + B + C + D) if (A + B + C + D) != 0 else 0  
    fom = round(fom,6)
    return [fom,A,B,C,D]


def calculate_metrics(S_true, S_pred,unique_classes):   
    overall_accuracy = accuracy_score(S_true, S_pred) 
    overall_accuracy=round(overall_accuracy,6)    
    precision = precision_score(S_true, S_pred, average=None, labels=unique_classes)
    recall = recall_score(S_true, S_pred, average=None, labels=unique_classes)
    f1 = f1_score(S_true, S_pred, average=None, labels=unique_classes)        
    conf_matrix = confusion_matrix(S_true, S_pred, labels=unique_classes)
    class_labels = pd.Series(unique_classes, name='Class')
    confusion_df = pd.DataFrame(conf_matrix, columns=[f'Pred_{cls}' for cls in unique_classes])
    confusion_df = pd.concat([class_labels, confusion_df], axis=1)
    metrics_df = pd.DataFrame({
        'Class': unique_classes,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    })    
    merged_df = pd.merge(metrics_df, confusion_df, on='Class', how='inner')
    merged_df.reset_index()
    merged_df.set_index('Class', inplace=True)
    return overall_accuracy, merged_df

def calculate_metrics_from_conf_matrix(conf_matrix,unique_classes):   

    cm = np.array(conf_matrix)
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision[i] =round( tp / (tp + fp) if (tp + fp) != 0 else 0,6)
        recall[i] = round(tp / (tp + fn) if (tp + fn) != 0 else 0,6)
        f1_score[i] =round( 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0,6)

    total_tp = np.sum(np.diag(cm))
    total_fp = np.sum(np.triu(cm, 1))
    total_fn = np.sum(np.tril(cm, -1))

    total_precision =round(total_tp / (total_tp + np.sum(total_fp)) if (total_tp + np.sum(total_fp)) != 0 else 0,6)
    total_recall = round(total_tp / (total_tp + np.sum(total_fn)) if (total_tp + np.sum(total_fn)) != 0 else 0,6)
    total_f1 = round(2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) != 0 else 0,6)

    overall_accuracy = total_tp / np.sum(cm)

    total_sum = np.sum(cm)
    p0 = total_tp / total_sum
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total_sum ** 2)
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 0

    metrics = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1': f1_score        
    }, index=unique_classes)

    for i, cls in enumerate(unique_classes):
        metrics[f'{cls}_true'] = cm[:, i]   
  
    return metrics,round(overall_accuracy,6),round(kappa,6),round(total_precision,6),round(total_recall,6),round(total_f1,6)



