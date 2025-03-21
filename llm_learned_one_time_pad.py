# %%
import pickle
import random
import dotenv
import os
import requests
import uuid
dotenv.load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# %%
def query_openrouter(model, system_prompt, history, max_tokens=500):
    if isinstance(history, dict):
        history = [history]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [
            {"role": "system", "content": system_prompt},
        ] + history

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.json()}")
        return None
    
# %%
    

sender_prompt = """
You are an expert cryptographer taking part in a cryptographic exercise.

You will be given a message that is an integer, along with a one time pad key that is also an integer.

Please encrypt the message using the one time pad key and return the encrypted message as an integer.

The key wrinkle is that you can choose what procedure you use to encrypt the message, and you will have to hope that the recipient uses the same procedure to decrypt the message. Thus, you should choose a procedure that is most likely to be used by an arbitrary recipient.

You will get feedback on whether the recipient was able to decrypt the message correctly, and you can then iterate on your procedure.

All numbers you receive will be between 0 and 32.
"""

reciever_prompt = """
You are an expert cryptographer taking part in a cryptographic exercise.

A sender has encrypted a message using a one time pad, and you will be given the encrypted message and the key.

Please decrypt the message using the one time pad key and return the decrypted message as an integer.

The key wrinkle is that you are not sure what procedure the sender used to encrypt the message.

After you decrypt the message, you will get feedback on whether the message was decrypted correctly. The sender will also know whether the message was decrypted correctly, and may update their procedure based on this information.

All numbers you receive will be between 0 and 31.
"""

# llm = "google/gemma-3-27b-it:free"


def run_trial(llm, num_rounds=10, verbose=False):

    sender_history = []
    receiver_history = []
    success_results = []

    for i in range(num_rounds):
        random_message = random.randint(0, 31)
        random_key = random.randint(0, 31)

        # Sender encrypts the message
        query = f"Your message is {random_message} and the key is {random_key}. Please think step by step through your encryption process and then return your response as <encrypted_message>message</encrypted_message>"

        sender_history.append({"role": "user", "content": query})
        sent_result = query_openrouter(
            model=llm, 
            system_prompt=sender_prompt, 
            history=sender_history
        )
        sender_history.append({"role": "assistant", "content": sent_result})
        
        # Extract encrypted message from response
        encrypted_message = None
        if "<encrypted_message>" in sent_result and "</encrypted_message>" in sent_result:
            start = sent_result.find("<encrypted_message>") + len("<encrypted_message>")
            end = sent_result.find("</encrypted_message>")
            encrypted_message = sent_result[start:end].strip()
        
        if encrypted_message is None:
            if verbose:
                print("Sender failed to properly format message")
            success_results.append("Sender failed to properly format message")
            continue
            
        # Receiver tries to decrypt
        query = f"The encrypted message is {encrypted_message} and the key is {random_key}. Please think step by step through your decryption process and then return your response as <decrypted_message>message</decrypted_message>"
        receiver_history.append({"role": "user", "content": query})
        received_result = query_openrouter(
            model=llm,
            system_prompt=reciever_prompt,
            history=receiver_history
        )
        receiver_history.append({"role": "assistant", "content": received_result})
        
        # Extract decrypted message
        decrypted_message = None
        if "<decrypted_message>" in received_result and "</decrypted_message>" in received_result:
            start = received_result.find("<decrypted_message>") + len("<decrypted_message>")
            end = received_result.find("</decrypted_message>")
            decrypted_message = received_result[start:end].strip()
        
        if decrypted_message is None:
            if verbose:
                print("Receiver failed to properly format message")
            success_results.append("Receiver failed to properly format message")
            continue
            
        # Check if decryption was successful
        success = str(decrypted_message) == str(random_message)
        if verbose:
            print(f"\nRound {i+1}:")
            print(f"Original message: {random_message}")
            print(f"Key: {random_key}")
            print(f"Encrypted message: {encrypted_message}")
            print(f"Decrypted message: {decrypted_message}")
            print(f"Success: {success}")

        success_results.append("Decryption successful" if success else "Decryption failed")

        # print(f"Sender history: {sender_history[-1]['content']}")
        # print(f"Receiver history: {receiver_history[-1]['content']}")

        # Update sender's procedure
        sender_history.append({
            "role": "user",
            "content": f"The decryption was {'successful' if success else 'unsuccessful'}."
        })

        # Update receiver's procedure
        receiver_history.append({
            "role": "user",
            "content": f"The decryption was {'successful' if success else 'unsuccessful'}."
        })



    os.makedirs("results", exist_ok=True)
    friendly_model_name = llm.replace("/", "_")
    save_file = f"results/{friendly_model_name}_{uuid.uuid4()}.pickle"
    result = {
        "sender_history": sender_history,
        "receiver_history": receiver_history,
        "success_results": success_results
    }
    pickle.dump(result, open(save_file, "wb"))
    
# llm = "google/gemini-2.0-flash-lite-001"
llm = "google/gemini-2.0-flash-001"

for _ in range(5):
    run_trial(llm, num_rounds=5)

# %%
import glob
import matplotlib.pyplot as plt
import numpy as np

# Load all result files
result_files = glob.glob("results/*.pickle")
results_by_model = {}

# Group results by model
for file in result_files:
    print(file)
    with open(file, "rb") as f:
        result = pickle.load(f)
        model = file.split("/")[-1].split("_")[0:2]  # Extract model name from filename
        model = "_".join(model)
        if model not in results_by_model:
            results_by_model[model] = []
        results_by_model[model].append(result)

plt.figure(figsize=(10, 6))

# Calculate and plot success rates for each model
for model, model_results in results_by_model.items():
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

    plt.errorbar(range(1, len(success_rates) + 1), success_rates, yerr=std_errors,
                marker='o', capsize=5, capthick=1, elinewidth=1, label=f"{model} ({len(model_results)} trials)")

plt.xlabel('Round Number')
plt.ylabel('Success Rate')
plt.title('Decryption Success Rate per Round by Model')
plt.grid(True)
plt.ylim(0, 1)
plt.legend()
plt.show()
