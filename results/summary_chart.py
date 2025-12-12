import pandas as pd                 #type: ignore
import seaborn as sns               #type: ignore
import matplotlib.pyplot as plt     #type: ignore

# --- Configuration ---
INPUT_FILE = "summary_report.csv"
OUTPUT_IMAGE = "bias_averages_chart.png"

def plot_averaged_bias(file_path):
    # 1. Load the data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    # Check how many runs we are averaging
    num_runs = df['filename'].nunique()
    print(f"Averaging data across {num_runs} distinct run(s).")

    # 2. Reshape the data (Melt)
    df_melted = df.melt(
        id_vars=['filename', 'model'], 
        value_vars=['positive_bias_effect', 'negative_bias_effect'],
        var_name='Bias Type', 
        value_name='Effect Size (%)'
    )

    # 3. Clean up labels
    df_melted['Bias Type'] = df_melted['Bias Type'].replace({
        'positive_bias_effect': 'Positive Bias', 
        'negative_bias_effect': 'Negative Bias'
    })

    # 4. Create the Chart
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Barplot automatically calculates the MEAN of the multiple runs
    # The error bars (vertical lines) show the 95% confidence interval by default
    chart = sns.barplot(
        data=df_melted,
        x="model",
        y="Effect Size (%)",
        hue="Bias Type",
        palette={"Positive Bias": "#2ecc71", "Negative Bias": "#e74c3c"},
        capsize=0.05, # Adds little caps to the error bars
        errorbar=('ci', 95) # Show 95% confidence interval
    )

    # 5. Customize Layout
    plt.title(f'Average Agreement Bias by Model (Aggregated over 5 Runs)', fontsize=14, pad=20)
    plt.axhline(0, color='black', linewidth=1)
    
    # Move legend to a nice spot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Add values on top of bars (Optional, helps with reading exact averages)
    for container in chart.containers:
        chart.bar_label(container, fmt='%.1f', padding=3)

    plt.tight_layout()

    # 6. Save
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Chart saved to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_averaged_bias(INPUT_FILE)