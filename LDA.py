import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def perform_lda(dataPath="data.csv"):
    """
    Perform Linear Discriminant Analysis using numeric features.
    
    Args:
        dataPath (str): Path to the CSV data file
    Returns:
        tuple: (accuracy, predictions, actual_values)
    """
    # Load the dataset
    data = pd.read_csv(dataPath)
    
    # Separating features from target
    label_column = 'Label'
    features = data.drop(columns=[label_column]).select_dtypes(include=['float64', 'int64'])
    labels = data[label_column]
    
    # Show excluded columns
    excluded_columns = set(data.columns) - set(features.columns)
    print("Excluded columns (including label):", excluded_columns)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split using the features DataFrame
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.3, random_state=42
    )

    # Initialize and fit LDA model
    lda = LDA()
    lda.fit(X_train, y_train)

    # Print LDA components information
    print("\nLDA Components Information:")
    print(f"Number of components: {lda.n_components}")
    print("\nExplained variance ratio:")
    print(lda.explained_variance_ratio_)
    
    # Print component coefficients
    print("\nLinear Discriminant Coefficients:")
    for i, component in enumerate(lda.coef_):
        print(f"\nLD{i+1} coefficients:")
        for feat, coef in zip(features.columns, component):
            print(f"{feat}: {coef:.4f}")

    # Make predictions
    y_pred = lda.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    print(f"Number of features: {len(features.columns)}")
    print(f"Number of unique classes: {len(labels.unique())}")
    print(f"Number of LDA components: {lda.n_components}")
        
    return accuracy, y_pred, y_test, lda

if __name__ == "__main__":
    perform_lda()