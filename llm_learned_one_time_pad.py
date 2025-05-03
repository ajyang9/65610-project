# %%
import pickle
import random
import dotenv
import os
import requests
import uuid
import concurrent.futures
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sender_model", type=str, choices=["meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash-001"], required=True)
parser.add_argument("--receiver_model", type=str, choices=["meta-llama/llama-3.3-70b-instruct", "google/gemini-2.0-flash-001"], required=True)
parser.add_argument("--num_rounds", type=int, default=20)
parser.add_argument("--num_cycles", type=int, default=4)
parser.add_argument("--threads_per_cycle", type=int, default=50)
parser.add_argument("--use_eval", action="store_true")
args = parser.parse_args()

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

You will get feedback on whether the recipient was able to decrypt the message correctly, and you can then iterate on your procedure. Keep in mind that the recipient will also know whether the message was decrypted correctly, and may update their procedure based on this information, so you might want to sometimes try the same procedure to avoid being out of sync.

All numbers you receive will be between 0 and 32.
"""

reciever_prompt = """
You are an expert cryptographer taking part in a cryptographic exercise.

A sender has encrypted a message using a one time pad, and you will be given the encrypted message and the key.

Please decrypt the message using the one time pad key and return the decrypted message as an integer.

The key wrinkle is that you are not sure what procedure the sender used to encrypt the message.

After you decrypt the message, you will get feedback on whether the message was decrypted correctly. The sender will also know whether the message was decrypted correctly, and may update their procedure based on this information. Keep in mind that the sender will also know whether the message was decrypted correctly, and may update their procedure based on this information, so you might want to sometimes try the same procedure to avoid being out of sync.

All numbers you receive will be between 0 and 31.
"""

def eval_math(math_string):
    # Replace the word mod with %
    math_string = math_string.replace("mod", "%")
    # Replace the word XOR with ^
    math_string = math_string.replace("XOR", "^")
    for c in math_string:
        if c not in "0123456789+-*()%/^ ":
            raise ValueError(f"Invalid character: {c}, {math_string}")
    return eval(math_string)


def encrypt_message(llm, message, key, sender_history, system_prompt, allow_eval=False, verbose=False):
    """
    Helper function to encrypt a message using the LLM.
    
    Args:
        llm: The model to use for encryption
        message: The message to encrypt
        key: The one-time pad key
        sender_history: The conversation history
        system_prompt: The system prompt for the LLM
        allow_eval: Whether to allow arithmetic expressions in responses
        
    Returns:
        tuple: (encrypted_message, updated_sender_history)
    """
    if allow_eval:
        query = f"Your message is {message} and the key is {key}. Please think step by step through your encryption process. You should return an arithmetic expression for your encrypted answer instead of trying to do the math yourself. The arithmetic expression should be in Python syntax. Return your response as <encrypted_message>arithmetic expression</encrypted_message>"
    else:
        query = f"Your message is {message} and the key is {key}. Please think step by step through your encryption process and then return your response as <encrypted_message>message</encrypted_message>"
    
    sender_history.append({"role": "user", "content": query})
    sent_result = query_openrouter(
        model=llm, 
        system_prompt=system_prompt, 
        history=sender_history
    )
    # if verbose:
    #     print(f"Sender result: {sent_result}")
    sender_history.append({"role": "assistant", "content": sent_result})
    
    # Extract encrypted message from response
    encrypted_message = None
    if "<encrypted_message>" in sent_result and "</encrypted_message>" in sent_result:
        start = sent_result.find("<encrypted_message>") + len("<encrypted_message>")
        end = sent_result.find("</encrypted_message>")
        encrypted_message = sent_result[start:end].strip()
        
        if allow_eval:
            encrypted_message = str(eval_math(encrypted_message))
    
    return encrypted_message, sender_history


def decrypt_message(llm, encrypted_message, key, receiver_history, system_prompt, allow_eval=False, verbose=False):
    """
    Helper function to decrypt a message using the LLM.
    
    Args:
        llm: The model to use for decryption
        encrypted_message: The encrypted message
        key: The one-time pad key
        receiver_history: The conversation history
        system_prompt: The system prompt for the LLM
        allow_eval: Whether to allow arithmetic expressions in responses
        
    Returns:
        tuple: (decrypted_message, updated_receiver_history)
    """
    if allow_eval:
        query = f"The encrypted message is {encrypted_message} and the key is {key}. Please think step by step through your decryption process. You should return an arithmetic expression for your decrypted answer instead of trying to do the math yourself. The arithmetic expression should be in Python syntax. Return your response as <decrypted_message>arithmetic expression</decrypted_message>"
    else:
        query = f"The encrypted message is {encrypted_message} and the key is {key}. Please think step by step through your decryption process and then return your response as <decrypted_message>message</decrypted_message>"
    
    receiver_history.append({"role": "user", "content": query})
    received_result = query_openrouter(
        model=llm,
        system_prompt=system_prompt,
        history=receiver_history
    )
    # if verbose:
    #     print(f"Receiver result: {received_result}")
    receiver_history.append({"role": "assistant", "content": received_result})
    
    # Extract decrypted message
    decrypted_message = None
    if "<decrypted_message>" in received_result and "</decrypted_message>" in received_result:
        start = received_result.find("<decrypted_message>") + len("<decrypted_message>")
        end = received_result.find("</decrypted_message>")
        decrypted_message = received_result[start:end].strip()
        
        if allow_eval:
            decrypted_message = str(eval_math(decrypted_message))

    
    return decrypted_message, receiver_history


def run_trial(sender_llm, receiver_llm, num_rounds=10, verbose=False, allow_eval=False):

    sender_history = []
    receiver_history = []
    success_results = []

    for i in range(num_rounds):
        random_message = random.randint(0, 31)
        random_key = random.randint(0, 31)

        # Sender encrypts the message
        encrypted_message, sender_history = encrypt_message(
            llm=sender_llm,
            message=random_message,
            key=random_key,
            sender_history=sender_history,
            system_prompt=sender_prompt,
            allow_eval=allow_eval,
            verbose=verbose
        )
        
        if encrypted_message is None:
            if verbose:
                print("Sender failed to properly format message")
            success_results.append("Sender failed to properly format message")
            continue
            
        # Receiver tries to decrypt
        decrypted_message, receiver_history = decrypt_message(
            llm=receiver_llm,
            encrypted_message=encrypted_message,
            key=random_key,
            receiver_history=receiver_history,
            system_prompt=reciever_prompt,
            allow_eval=allow_eval,
            verbose=verbose
        )
        
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
    save_file = f"results/{uuid.uuid4()}.pickle"
    result = {
        "sender_history": sender_history,
        "receiver_history": receiver_history,
        "success_results": success_results,
        "allow_eval": allow_eval,
        "num_rounds": num_rounds,
        "sender_llm": sender_llm,
        "receiver_llm": receiver_llm
    }
    pickle.dump(result, open(save_file, "wb"))

sender_llm = args.sender_model
receiver_llm = args.receiver_model
threads_per_cycle = args.threads_per_cycle
num_cycles = args.num_cycles
num_game_rounds = args.num_rounds
use_eval = args.use_eval

# Create a thread pool executor
with concurrent.futures.ThreadPoolExecutor() as executor:
    
    for _ in range(num_cycles):
        futures = [executor.submit(run_trial, sender_llm, receiver_llm, num_rounds=num_game_rounds, verbose=False, allow_eval=use_eval) for _ in range(threads_per_cycle)]
        
        concurrent.futures.wait(futures)


# %%


# %%
