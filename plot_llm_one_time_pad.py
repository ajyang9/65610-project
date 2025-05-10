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
    
            
        short_sender_llm = result["sender_llm"].split('/')[-1].split('-')[0]
        short_receiver_llm = result["receiver_llm"].split('/')[-1].split('-')[0]
        model = f"{short_sender_llm}→{short_receiver_llm}" 
        allow_eval = result["allow_eval"]
        key = (model, allow_eval)
        if key not in results_by_model_and_eval:
            results_by_model_and_eval[key] = []
        results_by_model_and_eval[key].append(result)

# Separate results into same model and different model
same_model_results = {}
different_model_results = {}

for (model, allow_eval), model_results in results_by_model_and_eval.items():
    sender, receiver = model.split('→')
    if sender == receiver:
        same_model_results[(model, allow_eval)] = model_results
    else:
        different_model_results[(model, allow_eval)] = model_results

print("Same model pairs:")
for (model, allow_eval), model_results in same_model_results.items():
    print(model, allow_eval)
    print(len(model_results))

print("\nDifferent model pairs:")
for (model, allow_eval), model_results in different_model_results.items():
    print(model, allow_eval)
    print(len(model_results))

# Create a figure with two subplots side by side
fig, ax = plt.subplots(2, 1, figsize=(8, 10))
ax = ax.flatten()
ax1, ax2 = ax

plot_error_bar = True

# Define colors for each model
model_colors = {}
unique_models = sorted(set(model for (model, _) in results_by_model_and_eval.keys()))
for i, model in enumerate(unique_models):
    model_colors[model] = f"C{i}"  # Use matplotlib's default color cycle

# Function to plot success rates on a given axis
def plot_success_rates(ax, results_dict, title):
    # Sort the keys to ensure consistent order in the legend
    sorted_keys = sorted(results_dict.keys(), key=lambda x: (x[0], not x[1]))
    
    # Calculate and plot success rates for each model and eval setting
    for model, allow_eval in sorted_keys:
        model_results = results_dict[(model, allow_eval)]
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
        
        if plot_error_bar:
            ax.errorbar(range(1, len(success_rates) + 1), success_rates, yerr=std_errors,
                        marker='o', capsize=5, capthick=1, elinewidth=1, 
                        color=model_colors[model], linestyle=linestyle,
                        label=f"{model} {eval_label} ({len(model_results)} trials)")
        else:
            ax.plot(range(1, len(success_rates) + 1), success_rates,
                    marker='o', color=model_colors[model], linestyle=linestyle,
                    label=f"{model} {eval_label} ({len(model_results)} trials)")
    
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Success Rate')
    ax.set_title(title)
    ax.grid(True)
    ax.set_ylim(0, 1)
    ax.legend()

# Plot the two subplots
plot_success_rates(ax1, same_model_results, 'Decryption Success Rate - Same Model')
plot_success_rates(ax2, different_model_results, 'Decryption Success Rate - Different Models')

plt.tight_layout()
plt.show()

# %%

encryptions_map = {}

for file in result_files:
    with open(file, "rb") as f:
        result = pickle.load(f)
        if result["num_rounds"] != num_rounds_to_plot:
            continue
        
        sender_llm = result["sender_llm"].split('/')[-1].split('-')[0]
        receiver_llm = result["receiver_llm"].split('/')[-1].split('-')[0]
        use_eval = result["allow_eval"]
        if not use_eval:
            continue

        key = (sender_llm, receiver_llm)
        if key not in encryptions_map:
            encryptions_map[key] = []

        encryptions = []
        correct = []
        for i, message in enumerate(result["sender_history"]):
            content = message["content"]
            role = message["role"]
            if role != "assistant":
                continue
            if "<encrypted_message>" not in content:
                continue
            message_content = content.split("<encrypted_message>")[1].split("</encrypted_message>")[0]
            encryptions.append(message_content)
            correct.append(result["success_results"][len(encryptions) - 1] == "Decryption successful")
        encryptions_map[key].append((encryptions, correct))

# %%
patterns = {
    'xor': ['(A ^ B)', 'A ^ B'],
    'xor and mod': ['(A ^ B) % 32', '(A ^ B) % 33'],
    'add and mod': ['(A + B) % 32', '(A + B) % 33'],
    'add': ['(A + B)', 'A + B'],
    'subtract and mod': ['(A - B) % 32', '(A - B) % 33'],
    'subtract': ['(A - B)', 'A - B'],
}

def match_pattern(encryption):
    """Match an encryption to a pattern."""
    encryption = encryption.strip()
    
    # Create regex patterns for each variant
    for pattern_name, pattern_variants in patterns.items():
        for variant in pattern_variants:
            # Create a regex pattern by escaping special characters and replacing A and B with digit patterns
            regex_pattern = variant
            for char in ['(', ')', '+', '^', '%']:
                if char in regex_pattern:
                    regex_pattern = regex_pattern.replace(char, f'\\{char}')
            
            # Replace A and B with digit patterns
            regex_pattern = regex_pattern.replace('A', r'\d+').replace('B', r'\d+')
            
            # Check if the pattern is in the encryption
            import re
            if re.search(regex_pattern, encryption):
                return pattern_name
    
    print(encryption)
    return "other"

# Create a 2x2 grid for all plots with space for the legend on the right
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

# Lists to store data for the legend
all_labels = ["XOR", "XOR and MOD", "ADD and MOD", "SUBTRACT and MOD", "ADD", "SUBTRACT", "Other"]
all_artists = []

for idx, (key, encryptions) in enumerate(encryptions_map.items()):
    successful_encryptions = [[] for _ in range(20)]
    pattern_counts = [{"xor": 0, "xor and mod": 0, "add and mod": 0, "subtract and mod": 0, "add": 0, "subtract": 0, "other": 0} for _ in range(20)]
    
    for example_encryptions, example_corrects in encryptions:
        for i, (encryption, correct) in enumerate(zip(example_encryptions, example_corrects)):
            if correct:
                successful_encryptions[i].append(encryption)
                pattern = match_pattern(encryption)
                pattern_counts[i][pattern] += 1
    
    # Create a stacked area plot for pattern distribution in the appropriate subplot
    ax = axes[idx]
    rounds = list(range(1, len(pattern_counts) + 1))
    
    xor_counts = [counts["xor"] for counts in pattern_counts]
    xor_mod_counts = [counts["xor and mod"] for counts in pattern_counts]
    add_mod_counts = [counts["add and mod"] for counts in pattern_counts]
    subtract_mod_counts = [counts["subtract and mod"] for counts in pattern_counts]
    add_counts = [counts["add"] for counts in pattern_counts]
    subtract_counts = [counts["subtract"] for counts in pattern_counts]
    other_counts = [counts["other"] for counts in pattern_counts]
    
    # Store the artists from the stackplot for the legend
    artists = ax.stackplot(rounds, 
                [xor_counts, xor_mod_counts, add_mod_counts, subtract_mod_counts, add_counts, subtract_counts, other_counts],
                labels=all_labels,
                alpha=0.7)
    
    if idx == 0:  # Only store artists from the first plot for the legend
        all_artists = artists
    
    ax.set_xlabel('Round Number')
    ax.set_ylabel('Number of Successful Encryptions')
    ax.set_title(f'Encryption Strategy Evolution: {key[0]} → {key[1]}')
    ax.grid(True, linestyle='--', alpha=0.7)

# Create a single legend to the right of all subplots
fig.legend(all_artists, all_labels, loc='center right', bbox_to_anchor=(0.9, 0.2), title="Encryption Strategies")

# plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to make room for the legend
plt.show()

# %%

