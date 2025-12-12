import pandas as pd # type: ignore
import glob
import os

# --- Configuration ---
# 1. Point this to the folder containing your CSVs, or specific files
# Example pattern: look for any csv file starting with 'results_'
INPUT_PATTERN = "results_*.csv" 
SUMMARY_OUTPUT = "summary_report.csv"  # Name of the combined report

def analyze_and_collect(df, filename, collector_list):
    """
    Analyzes a single DataFrame, prints results, and adds data to a collector list.
    """
    print("\n" + "#"*50)
    print(f"FILE SUMMARY: {filename}")
    print("#"*50)
    
    # Check for required columns
    required_cols = ['model', 'agreed_neutral', 'agreed_positive', 'agreed_negative']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: {filename} is missing required columns: {required_cols}")
        return

    # Group by Model
    for model, group in df.groupby('model'):
        # --- Calculations ---
        count = len(group)
        base_rate = group['agreed_neutral'].mean() * 100
        pos_rate = group['agreed_positive'].mean() * 100
        neg_rate = group['agreed_negative'].mean() * 100
        
        pos_bias = pos_rate - base_rate
        neg_bias = neg_rate - base_rate

        # --- Print to Console ---
        print(f"\nModel: {model}")
        print(f"  Total Samples: {count}")
        print(f"  Natural Agreement (Baseline): {base_rate:.1f}%")
        print(f"  Positive Framing Agreement:   {pos_rate:.1f}%")
        print(f"  Negative Framing Agreement:   {neg_rate:.1f}%")
        print(f"  -> Positive Bias Effect: {pos_bias:+.1f}%")
        print(f"  -> Negative Bias Effect: {neg_bias:+.1f}%")

        # --- Add to Collector for CSV Export ---
        collector_list.append({
            'filename': filename,
            'model': model,
            'total_samples': count,
            'natural_agreement_pct': round(base_rate, 2),
            'positive_framing_pct': round(pos_rate, 2),
            'negative_framing_pct': round(neg_rate, 2),
            'positive_bias_effect': round(pos_bias, 2),
            'negative_bias_effect': round(neg_bias, 2)
        })
    
    print("-" * 50)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Find all files matching the pattern
    csv_files = ["subjective1/agreement_bias_results_summary.csv","subjective2/agreement_bias_results_summary.csv","subjective3/agreement_bias_results_summary.csv","subjective4/agreement_bias_results_summary.csv","subjective6/agreement_bias_results_summary.csv"]
    i = 0
    csv_files = []
    while i < 10:
        filestr = "subjective" + str(i) + "/agreement_bias_results_summary.csv"
        csv_files.append(filestr)
        i += 1
    if not csv_files:
        print(f"No files found matching pattern: {INPUT_PATTERN}")
    else:
        print(f"Found {len(csv_files)} file(s):")
        for f in csv_files:
            print(f" - {f}")

summary_data = []

if not csv_files:
    print(f"No files found matching pattern: {INPUT_PATTERN}")
else:
    print(f"Found {len(csv_files)} file(s). Processing...\n")
    
    # 2. Process each file
    for f in csv_files:
        try:
            df_current = pd.read_csv(f)
            analyze_and_collect(df_current, f, summary_data)
        except Exception as e:
            print(f"Could not process file {f}: {e}")

    # 3. Save the Combined Report
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(SUMMARY_OUTPUT, index=False)
        print(f"\nSuccessfully saved combined summary to: {SUMMARY_OUTPUT}")
        print(summary_df.head()) # Preview the summary table
    else:
        print("\nNo data was collected to save.")