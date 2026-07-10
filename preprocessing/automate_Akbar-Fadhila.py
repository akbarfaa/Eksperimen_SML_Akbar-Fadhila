import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """Load raw dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Clean and preprocess the input dataframe."""
    print("Starting preprocessing...")
    
    # Drop unique ID column
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Remove minor gender anomaly
    df = df[df['gender'] != 'Other']
    
    # Separate features and target if target exists
    has_target = 'stroke' in df.columns
    if has_target:
        X = df.drop(columns=['stroke'])
        y = df['stroke']
    else:
        X = df
        y = None
        
    # Define columns by type
    num_features = ['age', 'avg_glucose_level', 'bmi']
    cat_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    bin_features = ['hypertension', 'heart_disease']
    
    # Preprocessing pipelines
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        ('bin', 'passthrough', bin_features)
    ])
    
    # Fit and transform
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Reconstruct dataframe with encoded categorical column names
    encoded_cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
    all_feature_names = num_features + list(encoded_cat_features) + bin_features
    
    df_preprocessed = pd.DataFrame(X_preprocessed, columns=all_feature_names)
    
    if has_target:
        df_preprocessed['stroke'] = y.reset_index(drop=True)
        
    print("Preprocessing completed!")
    return df_preprocessed

def save_data(df, output_path):
    """Save the preprocessed dataframe to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    # Define directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "..", "namadataset_raw", "healthcare-dataset-stroke-data.csv")
    output_data_path = os.path.join(base_dir, "healthcare-dataset-stroke-data_preprocessing", "stroke_data_preprocessed.csv")
    
    try:
        df_raw = load_data(raw_data_path)
        df_prep = preprocess_data(df_raw)
        save_data(df_prep, output_data_path)
        print("Preprocessed data shape:", df_prep.shape)
    except Exception as e:
        print("Error during preprocessing:", str(e))

