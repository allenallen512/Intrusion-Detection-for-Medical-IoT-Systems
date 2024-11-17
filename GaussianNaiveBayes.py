import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

def GaussianNaiveBayes():
    data = pd.read_csv("data.csv")

    # drop non numeric columns and label
    X = data.select_dtypes(include=['float64', 'int64']).drop('Label', axis=1)
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    results = GaussianNB()
    results.fit(X_train, y_train)

    y_pred = results.predict(X_test)

    confusion = confusion_matrix(y_test, y_pred)

    # Print Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    tn, fp, fn, tp = confusion.ravel()

    print(f"Confusion matrix for Gaussian Naive Bayes:")
    print("-" * 40)
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("-" * 40)
    print("\n")

GaussianNaiveBayes()