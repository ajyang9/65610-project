# %%

import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load all result files
result_files = glob.glob("results/*.pickle")
results_by_model_and_eval = {}

num_rounds_to_plot = 20

# Group results by model and eval setting
for file in result_files:
    with open(file, "rb") as f:
        result = pickle.load(f)
        if result["num_rounds"] != num_rounds_to_plot:
            continue
        model = file.split("/")[-1].split("_")[0:2]  # Extract model name from filename
        model = "_".join(model)
        allow_eval = result["allow_eval"]
        key = (model, allow_eval)
        if key not in results_by_model_and_eval:
            results_by_model_and_eval[key] = []
        results_by_model_and_eval[key].append(result)

        # print(result)

for (model, allow_eval), model_results in results_by_model_and_eval.items():
    print(model, allow_eval)
    print(len(model_results))


plt.figure(figsize=(12, 8))

# Define colors for each model
model_colors = {}
unique_models = sorted(set(model for (model, _) in results_by_model_and_eval.keys()))
for i, model in enumerate(unique_models):
    model_colors[model] = f"C{i}"  # Use matplotlib's default color cycle

# Sort the keys to ensure consistent order in the legend
sorted_keys = sorted(results_by_model_and_eval.keys(), key=lambda x: (x[0], not x[1]))

# Calculate and plot success rates for each model and eval setting
for model, allow_eval in sorted_keys:
    model_results = results_by_model_and_eval[(model, allow_eval)]
    success_rates = []
    std_errors = []
    
    for round_idx in range(len(model_results[0]["success_results"])):
        round_successes = []
        for result in model_results:
            success = 1 if result["success_results"][round_idx] == "Decryption successful" else 0
            round_successes.append(success)
        success_rate = np.mean(round_successes)
        std_error = np.std(round_successes) / np.sqrt(len(round_successes))
        success_rates.append(success_rate)
        std_errors.append(std_error)

    eval_label = "with eval" if allow_eval else "without eval"
    linestyle = '-' if allow_eval else '--'
    
    plt.errorbar(range(1, len(success_rates) + 1), success_rates, yerr=std_errors,
                marker='o', capsize=5, capthick=1, elinewidth=1, 
                color=model_colors[model], linestyle=linestyle,
                label=f"{model} {eval_label} ({len(model_results)} trials)")

plt.xlabel('Round Number')
plt.ylabel('Success Rate')
plt.title('Decryption Success Rate per Round by Model and Eval Setting')
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.show()
# %%
