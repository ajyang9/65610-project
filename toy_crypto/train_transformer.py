import torch
import torch.nn as nn
from model import Model
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Categorical


def random_binary(shape):
    return torch.randint(0, 2, shape)

def index2onehot(x, num_classes):
    return torch.nn.functional.one_hot(x, num_classes=num_classes).float()

def onehot2index(x):
    return torch.argmax(x, dim=-1)

class Crypto(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        sk_len = model_config["sk_len"]
        message_len = model_config["message_len"]
        ct_len = model_config["ct_len"]
        num_layers = model_config["num_layers"]
        dim = model_config["dim"]
        num_heads = model_config["num_heads"]

        self.sender = Model(num_layers, dim, num_heads, sk_len + message_len, ct_len, 2)
        self.receiver = Model(num_layers, dim, num_heads, sk_len + ct_len, message_len, 2)
        self.adversary = Model(num_layers, dim, num_heads, ct_len, message_len, 2)
    
    def forward(self, sk, message, temperature=1.0):
        sk_onehot = index2onehot(sk, 2)
        message_onehot = index2onehot(message, 2)
        sk_message = torch.cat([sk_onehot, message_onehot], dim=1)
        ct_distribution = self.sender(sk_message) 
        ct_onehot = F.gumbel_softmax(ct_distribution.logits, tau=temperature, hard=True)
        ct = onehot2index(ct_onehot)

        sk_ct_onehot = torch.cat([sk_onehot, ct_onehot], dim=1)  
        rec_distribution = self.receiver(sk_ct_onehot)
        adv_distribution = self.adversary(ct_onehot)
        return ct, rec_distribution, adv_distribution
    
    def rec_loss(self, message, rec_distribution):
        return -rec_distribution.log_prob(message).mean()

    def adv_loss(self, message, adv_message):
        return -adv_message.log_prob(message).mean()
        
        
    
def train(model, model_config, sender_optim, receiver_optim, adversary_optim, train_config):
    batch_size = train_config["batch_size"]
    n_steps = train_config["n_steps"]
    eval_steps = train_config["eval_steps"]
    n_samples = train_config["n_samples"]

    sk_len = model_config["sk_len"]
    message_len = model_config["message_len"]

    rec_losses = []
    adv_losses = []
    losses = []

    pbar = tqdm(range(n_steps))
    for step in pbar:
        sk = random_binary((batch_size, sk_len))
        message = random_binary((batch_size, message_len))
        ct, rec_distribution, adv_distribution = model(sk, message)
        rec_loss = model.rec_loss(message, rec_distribution)
        rec_losses.append(rec_loss)
        adv_loss = model.adv_loss(message, adv_distribution)
        adv_losses.append(adv_loss)

        loss = rec_loss - adv_loss
    
        sender_optim.zero_grad()
        receiver_optim.zero_grad()
        adversary_optim.zero_grad()
        loss.backward()
        
        if step % 400 < 200:
            adversary_optim.step()
            sender_optim.step()
        else: 
            receiver_optim.step()
            sender_optim.step()
        
            

        pbar.set_description(f"loss: {loss:3f}, rec_loss: {rec_loss:3f}, adv_loss: {adv_loss:3f}")

        if step % eval_steps == 0:
            print("Samples: \n")
            sk = random_binary((n_samples, sk_len))
            message = random_binary((n_samples, message_len))
            ct, rec_distribution, adv_distribution = model(sk, message)
            rec_samples = rec_distribution.sample()
            adv_samples = adv_distribution.sample()
            for i in range(n_samples):
                print(f"sk: {sk[i]}")
                print(f"message: {message[i]}")
                print(f"ct: {ct[i]}")
                print(f"rec_message: {rec_samples[i]}")
                print(f"adv_message: {adv_samples[i]}")
                print()
    
    plt.plot(rec_losses, label="rec_loss")
    plt.plot(adv_losses, label="adv_loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model_config = {
        "sk_len": 1,
        "message_len": 1,
        "ct_len": 1,
        "num_layers": 2,
        "dim": 32,
        "num_heads": 4,
    }
    train_config = {
        "batch_size": 512,
        "n_steps": 20000,
        "eval_steps": 1000,
        "n_samples": 5,
    }
    model = Crypto(model_config)
    sender_optim = torch.optim.SGD(model.sender.parameters(), lr=1e-2)
    receiver_optim = torch.optim.SGD(model.receiver.parameters(), lr=1e-2)
    adversary_optim = torch.optim.SGD(model.adversary.parameters(), lr=1e-2, maximize=True)
    train(model, model_config, sender_optim, receiver_optim, adversary_optim, train_config)





