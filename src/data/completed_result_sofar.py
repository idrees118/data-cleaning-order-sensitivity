import pandas as pd
from glob import glob

# Function to merge all CSVs for a dataset
def merge_results(dataset_name):
    # Match all CSVs for this dataset
    files = glob(f'results/tables/{dataset_name}_scientific_results_*.csv')
    if not files:
        print(f"No files found for {dataset_name}")
        return
    # Read and concatenate
    df_list = [pd.read_csv(f) for f in files]
    merged_df = pd.concat(df_list, ignore_index=True)
    # Save to new CSV
    output_file = f'results/tables/{dataset_name}_completed_result_sofar.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(files)} files into {output_file}, total rows: {len(merged_df)}")

# Run for all datasets
for dataset in ['pima', 'adult', 'credit_default', 'home_credit']:
    merge_results(dataset)
