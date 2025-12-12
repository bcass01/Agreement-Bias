import pandas as pd # type: ignore

df = pd.read_csv('summary_report.csv')

gpt = df[df['model'] == 'gpt-4o']
claude = df[df['model'] == 'claude-4.5-sonnet']
llama = df[df['model'] == 'llama-3-70b']

models = [gpt, claude, llama]
print("Model, Natural, Positive Bias, Negative Bias")

for model in models:
    modelname = model.iloc[1]['model']
    average = model['natural_agreement_pct'].mean()
    pos = model['positive_bias_effect'].mean()
    neg = model['negative_bias_effect'].mean()
    print(modelname, ",", average, ",", pos, ",", neg)