import pandas as pd
import os

def create_balanced_sample(full_path, output_path, n_per_class=200):
    print(f"Reading full dataset from {full_path}...")
    df = pd.read_csv(full_path, header=None)
    
    # Identify the label column (last one)
    label_col = df.columns[-1]
    
    # Sample from each class
    balanced_df = pd.DataFrame()
    for label in df[label_col].unique():
        class_subset = df[df[label_col] == label]
        sample_size = min(len(class_subset), n_per_class)
        sampled = class_subset.sample(n=sample_size, random_state=42)
        balanced_df = pd.concat([balanced_df, sampled])
        print(f"Class {label}: Sampled {sample_size} rows")
    
    # Shuffle the final dataframe
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    balanced_df.to_csv(output_path, index=False, header=False)
    print(f"✅ Balanced sample saved to {output_path} ({len(balanced_df)} rows)")

if __name__ == "__main__":
    create_balanced_sample("/home/me/mini-proj/mitbih_train.csv", "/home/me/mini-proj/supervisor_notebooks/sample_mitbih.csv")
