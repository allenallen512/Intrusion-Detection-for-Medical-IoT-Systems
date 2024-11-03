from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

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


find_top_n_features(dataPath, 5)