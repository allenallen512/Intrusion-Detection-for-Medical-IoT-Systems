# Heavily derived from the SVC_Example.ipynb file on canvas

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def train_SVC():         # might have to change default n_estimators
    data = pd.read_csv("data.csv")

    # drop non numeric columns and label
    X = data.select_dtypes(include=['float64', 'int64']).drop('Label', axis=1)
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    kernelTypes = ['linear', 'rbf', 'poly']
    results = []
    confusion_matrices = []

    for kernelType in tqdm(kernelTypes, desc='Training'):
        model = SVC(kernel=kernelType, C=1)
        model.fit(X_train, y_train)

        y_pred=model.predict(X_test)

        confusion_matrices.append(confusion_matrix(y_test, y_pred))

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store results
        results.append({
            'Kernel': kernelType,
            'Accuracy': f"{accuracy:.2f}",
            'Precision': f"{precision:.2f}",
            'Recall': f"{recall:.2f}",
            'F1 Score': f"{f1:.2f}"
        })
    
    for i in range(len(kernelTypes)):
        tn, fp, fn, tp = confusion_matrices[i].ravel()

        print(f"Confusion matrix for kernel: {kernelTypes[i]}")
        print("-" * 40)
        print(f"True Positives:  {tp}")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print("-" * 40)
        print("\n")

    # Convert results to a DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    print("\nComparison of Kernel Performance:\n", results_df)

train_SVC()
