# Agreement-Bias
Measuring and mitigating agreement bias in LLMs.

# SETUP
Created the objective questions dataset by running truthful_prompts.py, which turned the TruthfulQA dataset into a series of neutral and framed prompts.
Created the subjective questions dataset by converting TableS1.csv (from the Moral Machine dataset) into neutral, positively-framed, and negatively-framed prompts using convert_scenario_csv.py

# EXPERIMENT
Ran generate_responses.py and generate_moral_responses.py to prompt LLMs with objective and subjective datasets, respectively. Responses were recorded in json files and then evaluated using evaluate_moral_results.py, which created a results csv file.

# RESULTS
To summarize all runs, we used summarize_moral_results.py, which generated summary_report.csv. These results and their averages were used to create the charts and tables seen in the paper.
