from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import pandas as pd

def confusion_matrix(y_true, y_predicted, labels):
    df = pd.DataFrame(data=sk_confusion_matrix(y_true, y_predicted), index=labels, columns=labels)
    df.index.name = 'true classes'
    df.columns.name = 'predicted classes'
    return df
