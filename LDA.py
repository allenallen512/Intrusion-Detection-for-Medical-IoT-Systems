import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

dataPath="data.csv", 
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

def perform_lda(features_scaled, labels):
  
    #put in the features scaled and the labels we want to predict
    
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
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Generate classification report
    class_report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(class_report)
    accuracy = accuracy_score(y_test, y_pred)   
    return conf_matrix, class_report, accuracy
    

if __name__ == "__main__":
    perform_lda()