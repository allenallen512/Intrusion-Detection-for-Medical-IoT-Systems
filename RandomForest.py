import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_random_forest(n_estimators):
    print(f"Training Random Forest with {n_estimators} estimators")
    
    data = pd.read_csv("data.csv")

    # drop non numeric columns and label
    X = data.select_dtypes(include=['float64', 'int64']).drop('Label', axis=1)
    y = data['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_classifier.predict(X_test)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix Results:")
    print("-" * 40)
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print("-" * 40)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_classifier.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    print("\nTop 5 Most Important Features:")
    print(feature_importance.head(5))

    # Calculate and print accuracy
    accuracy = rf_classifier.score(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy:.2%}")


train_random_forest(n_estimators=150)
