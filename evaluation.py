from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, features, labels):
    predictions = model.predict(features)

    acc = accuracy_score(labels, predictions)
    rec = recall_score(labels, predictions, average='macro')
    pre = precision_score(labels, predictions, average='macro')
    f1_ = f1_score(labels, predictions, average='macro')
    cfm = confusion_matrix(labels, predictions, normalize="true")


    return {
        'Accuracy' : acc,
        'Recall'   : rec,
        'Precision': pre,
        'f1' : f1_,
        'Confusion Matrix' : cfm
    }
