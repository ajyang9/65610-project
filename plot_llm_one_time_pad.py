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
        
        # Only include results where sender and receiver are the same model
        if result["sender_llm"] != result["receiver_llm"]:
            continue
            
        model = result["sender_llm"]  # Use the actual model name from the result
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

# Create a plot for different sender-receiver pairs
plt.figure(figsize=(12, 8))

# Group results by sender-receiver pair and eval setting
results_by_pair_and_eval = {}
for result_file in result_files:
    with open(result_file, 'rb') as f:
        result = pickle.load(f)
        
    sender_model = result["sender_llm"]
    receiver_model = result["receiver_llm"]
    allow_eval = result["allow_eval"]
    pair_key = (sender_model, receiver_model, allow_eval)

    if sender_model == receiver_model:
        continue
    
    if pair_key not in results_by_pair_and_eval:
        results_by_pair_and_eval[pair_key] = []
    
    results_by_pair_and_eval[pair_key].append(result)

# Print the number of results for each pair
for (sender, receiver, allow_eval), pair_results in results_by_pair_and_eval.items():
    print(f"Sender: {sender}, Receiver: {receiver}, Allow Eval: {allow_eval}")
    print(f"Number of trials: {len(pair_results)}")

# Define colors for each sender-receiver pair
pair_colors = {}
unique_pairs = sorted(set((sender, receiver) for (sender, receiver, _) in results_by_pair_and_eval.keys()))
for i, pair in enumerate(unique_pairs):
    pair_colors[pair] = f"C{i}"  # Use matplotlib's default color cycle

# Sort the keys to ensure consistent order in the legend
sorted_pair_keys = sorted(results_by_pair_and_eval.keys(), key=lambda x: (x[0], x[1], not x[2]))

# Calculate and plot success rates for each sender-receiver pair and eval setting
for sender, receiver, allow_eval in sorted_pair_keys:
    pair_results = results_by_pair_and_eval[(sender, receiver, allow_eval)]
    success_rates = []
    std_errors = []
    
    for round_idx in range(len(pair_results[0]["success_results"])):
        round_successes = []
        for result in pair_results:
            success = 1 if result["success_results"][round_idx] == "Decryption successful" else 0
            round_successes.append(success)
        success_rate = np.mean(round_successes)
        std_error = np.std(round_successes) / np.sqrt(len(round_successes))
        success_rates.append(success_rate)
        std_errors.append(std_error)

    sender_short = sender.split('/')[-1].split('-')[0]  # Simplify model names
    receiver_short = receiver.split('/')[-1].split('-')[0]
    eval_label = "with eval" if allow_eval else "without eval"
    linestyle = '-' if allow_eval else '--'
    
    plt.errorbar(range(1, len(success_rates) + 1), success_rates, yerr=std_errors,
                marker='o', capsize=5, capthick=1, elinewidth=1, 
                color=pair_colors[(sender, receiver)], linestyle=linestyle,
                label=f"{sender_short}→{receiver_short} {eval_label} ({len(pair_results)} trials)")

plt.xlabel('Round Number')
plt.ylabel('Success Rate')
plt.title('Decryption Success Rate per Round by Sender-Receiver Pair and Eval Setting')
plt.grid(True)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

