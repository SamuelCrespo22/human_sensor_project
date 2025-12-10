from config import RANDOM_STATE
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.metrics as metrics

# ========================
# Data Loading Functions
# ========================

def load_iris_data():
    return load_iris(as_frame=True)

def load_ds_data():
    return pd.read_csv('feature_set.csv')

# ========================
# Data Splitting Functions
# ========================

def train_test(X, y, test_size=0.2, random_state=RANDOM_STATE):
    X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_validation_test(X, y, test_size=0.2, val_size=0.25, shuffle=True, random_state=RANDOM_STATE):
    X_temp, X_test, y_temp, y_test = ms.train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=shuffle
    )
    X_train, X_val, y_train, y_val = ms.train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp, shuffle=shuffle
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def k_fold_split(X, y, n_splits=5, shuffle=True, random_state=RANDOM_STATE):
    kf = ms.KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        yield X_train, X_test, y_train, y_test

# ========================
# Accuracy Metrics
# ========================

def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred)

def recall_score(y_true, y_pred, average='macro'):
    return metrics.recall_score(y_true, y_pred, average=average, zero_division=0)

def precision_score(y_true, y_pred, average='macro'):
    return metrics.precision_score(y_true, y_pred, average=average, zero_division=0)

def f1_score(y_true, y_pred, average='macro'):
    return metrics.f1_score(y_true, y_pred, average=average, zero_division=0)

def show_accuracy_metrics(y_true, y_pred, model_name, show=True):
    # Flatten y_true if it's a pandas DataFrame/Series, as sklearn expects 1D arrays
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true_flat = y_true.values.ravel()
    else:
        y_true_flat = np.asarray(y_true).ravel()

    # y_pred from sklearn classifiers is a numpy array, already flat.
    y_pred_flat = np.asarray(y_pred).ravel()
    
    # Get all unique labels from both true and predicted values to build the matrix correctly
    all_labels = np.unique(np.concatenate((y_true_flat, y_pred_flat)))
    
    # Create confusion matrix using all labels to ensure consistent shape
    cm = metrics.confusion_matrix(y_true, y_pred, labels=all_labels)

    result = {
        'Model': model_name,
        'Confusion Matrix': cm,
        'Precision Score': precision_score(y_true, y_pred),
        'Recall Score': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred)
    }

    if show:
        # Display the confusion matrix as a labeled pandas DataFrame for clarity
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
        cm_df.index.name = 'True Label'
        cm_df.columns.name = 'Predicted Label'
        print(f"\n=== {model_name} ===")
        print("Precision: {:.3f}".format(result['Precision Score']))
        print("Recall:    {:.3f}".format(result['Recall Score']))
        print("F1 Score:  {:.3f}".format(result['F1 Score']))
        print("\nConfusion Matrix:")
        print(cm_df)
    return result

def pretty_print_cm_matrix(cm, labels, title="Confusion Matrix"):
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.index.name = "True"
    df_cm.columns.name = "Predicted"

    print(f"\n{title}:")
    print(df_cm)

# ========================
# Baseline Classifiers
# ========================

def random_classifier(X_train, y_train, X_test):
    clf = DummyClassifier(strategy="uniform", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def oner_class_classifier(X_train, y_train, X_test):
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def knn_classifier(X_train, y_train, X_test, n_neighbors=3):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred