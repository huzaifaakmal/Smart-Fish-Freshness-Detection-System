import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating EDA reports from {csv_path}...")
    
    # 1. Species Distribution (Top 20)
    plt.figure(figsize=(12, 8))
    species_counts = df['species_name'].value_counts()
    sns.barplot(x=species_counts.head(20).values, y=species_counts.head(20).index)
    plt.title('Top 20 Species Distribution')
    plt.xlabel('Number of Samples')
    plt.ylabel('Species Name')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'species_distribution_top20.png'))
    plt.close()
    
    # 2. Environment Distribution
    plt.figure(figsize=(10, 6))
    env_counts = df['environment'].value_counts()
    plt.pie(env_counts.values, labels=env_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Environment/Condition Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, 'environment_distribution.png'))
    plt.close()
    
    # 3. Samples per Species Statistics
    plt.figure(figsize=(10, 6))
    sns.histplot(species_counts.values, bins=20, kde=True)
    plt.title('Samples per Species Distribution')
    plt.xlabel('Number of Samples')
    plt.ylabel('Frequency (Number of Species)')
    plt.savefig(os.path.join(output_dir, 'samples_per_species_hist.png'))
    plt.close()
    
    # 4. Species per Environment
    plt.figure(figsize=(12, 6))
    env_species = df.groupby('environment')['species_name'].nunique()
    sns.barplot(x=env_species.index, y=env_species.values)
    plt.title('Number of Unique Species per Environment')
    plt.ylabel('Unique Species Count')
    plt.savefig(os.path.join(output_dir, 'unique_species_per_env.png'))
    plt.close()

    print(f"EDA charts saved to {output_dir}")
    
    # Print summary stats
    print("\nSummary Statistics:")
    print(f"Total Samples: {len(df)}")
    print(f"Total Unique Species: {df['species_name'].nunique()}")
    print("\nSamples per Environment:")
    print(env_counts)
    print("\nTop 5 Species:")
    print(species_counts.head(5))

if __name__ == "__main__":
    CSV_PATH = "dataset/fish_metadata.csv"
    OUTPUT_DIR = "notebooks/eda_results"
    run_eda(CSV_PATH, OUTPUT_DIR)
