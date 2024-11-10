from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataPath = "data.csv"

def find_top_n_features(file_path, n):
    data = pd.read_csv(file_path)
    
    # Separating features from target
    label_column = 'Label'  # specify your label column name
    features = data.drop(columns=[label_column]).select_dtypes(include=['float64', 'int64'])
    
    # Add this to see excluded columns
    excluded_columns = set(data.columns) - set(features.columns)
    print("Excluded columns (including label):", excluded_columns)

    # Standardizing the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(features)
    
    # Applying PCA
    pca = PCA(n_components=n)
    pca.fit(data_scaled)
    
    # Getting the top n features based on explained variance
    components = pca.components_
    #doesprint(components)
    explained_variance = pca.explained_variance_ratio_
    
    # Creating a DataFrame of feature importance
    feature_importance = pd.DataFrame()
    for i in range(n):
        feature_importance[f'PC{i+1}'] = abs(components[i])
    
    feature_importance.index = features.columns
    
    # Getting overall importance by summing across components
    feature_importance['Overall_Importance'] = feature_importance.sum(axis=1)
    feature_importance = feature_importance.sort_values('Overall_Importance', ascending=False)
    
    # Add this to see the full table
    print("\nFeature Importance by Principal Component:")
    pd.set_option('display.float_format', lambda x: '%.3f' % x)  # Format to 3 decimal places
    print(feature_importance)
    
    # Displaying results
    print(f"Total explained variance ratio: {sum(explained_variance):.2%}")
    print("\nExplained variance ratio by component:")
    for i, var in enumerate(explained_variance):
        print(f"PC{i+1}: {var:.2%}")
    
    print("\nTop {n} most important features:")
    print(feature_importance['Overall_Importance'].head(n))
    
    # After PCA is applied, add this before the final return:
    print("\nPrincipal Component Linear Combinations:")
    pc_composition = pd.DataFrame(
        components.T,
        columns=[f'PC{i+1}' for i in range(n)],
        index=features.columns
    )
    
    for pc in range(n):
        print(f"\nPC{pc+1} = ", end="")
        # Get the components for this PC
        coefficients = components[pc]
        terms = []
        for feat, coef in zip(features.columns, coefficients):
            if abs(coef) > 0.1:  # Only show significant contributions (adjust threshold if needed)
                terms.append(f"({coef:.3f} × {feat})")
        print(" + ".join(terms))
    
    return feature_importance


find_top_n_features(dataPath, 2)

def perform_kmeans_clustering(dataPath, n_components=2):
    print("\nPerforming K-means clustering on PCA components...")
    
    # Read the data
    data = pd.read_csv(dataPath)
    features = data.select_dtypes(include=['float64', 'int64']).drop('Label', axis=1)
    labels = data['Label']
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pc_features = pca.fit_transform(features)
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(
        data=pc_features,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        pca_df, labels, test_size=0.1, random_state=42
    )
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    
    # Make predictions
    y_pred = kmeans.predict(X_test)
    
    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    # Calculate percentages
    tn_pct = (tn/total) * 100
    fp_pct = (fp/total) * 100
    fn_pct = (fn/total) * 100
    tp_pct = (tp/total) * 100       
    
    print(f"""
        True Negatives: {tn} ({tn_pct:.2f}%)
        False Positives: {fp} ({fp_pct:.2f}%)
        False Negatives: {fn} ({fn_pct:.2f}%)
        True Positives: {tp} ({tp_pct:.2f}%)
        """)
    
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # # Visualize the clusters
    # plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(X_test['PC1'], X_test['PC2'], c=y_pred, cmap='viridis')
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    # plt.title('K-means Clustering Results on PCA Components')
    # plt.colorbar(scatter)
    # plt.show()
    # # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")

# Test the clustering
# perform_kmeans_clustering("data.csv")
