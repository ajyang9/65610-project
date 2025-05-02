import torch
import torch.nn as nn
from model import MLP
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def random_binary(shape, ratio=0.5):
    return torch.bernoulli(torch.full(shape, ratio)).float()

def target_ct(sk, message):
    return (torch.round((sk + message)) % 2).float()

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

class Crypto(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        sk_dim = model_config["sk_dim"]
        message_dim = model_config["message_dim"]
        ct_dim = model_config["ct_dim"]
        hidden_dim = model_config["hidden_dim"]
        depth = model_config["depth"]

        self.sender = MLP(sk_dim + message_dim, ct_dim, hidden_dim, depth)
        self.receiver = MLP(sk_dim + ct_dim, message_dim, hidden_dim, depth)
        self.adversary = MLP(ct_dim, message_dim, hidden_dim, depth)
        self.quantize = StraightThroughEstimator()
    
    def forward(self, sk, message):
        sk_message = torch.cat([sk, message], dim=1)
        ct = self.quantize(self.sender(sk_message))
        ct = target_ct(sk, message)
        sk_ct = torch.cat([sk, ct], dim=1)
        rec_message = self.quantize(self.receiver(sk_ct))
        adv_message = self.quantize(self.adversary(ct))
        return ct, rec_message, adv_message
    
    def rec_loss(self, message, rec_message):
        return F.mse_loss(rec_message, message)

    def adv_loss(self, message, adv_message):
        return F.mse_loss(adv_message, message)

def get_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, params):
    i = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(params[i:i+numel].view(p.size()))
        i += numel

def loss_landscape(model, sk, message):
    # plot loss landscape against self.sender.parameters and self.receiver.parameters
    def get_loss(sender_params, receiver_params):
        set_params(model.sender, sender_params)
        set_params(model.receiver, receiver_params)
        ct, rec_message, adv_message = model(sk, message)
        rec_loss = model.rec_loss(message, rec_message)
        adv_loss = model.adv_loss(message, adv_message)
        loss = rec_loss - adv_loss
        return loss.item()
    
    init_sender_params = get_params(model.sender)
    init_receiver_params = get_params(model.receiver)
    sender_dir = torch.randn_like(init_sender_params)
    receiver_dir = torch.randn_like(init_receiver_params)
    sender_dir /= torch.norm(sender_dir)
    receiver_dir /= torch.norm(receiver_dir)

    alphas = np.linspace(-5, 5, 100)
    betas = np.linspace(-5, 5, 100)
    losses = np.zeros((len(alphas), len(betas)))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            sender_params = init_sender_params + alpha * sender_dir
            receiver_params = init_receiver_params + beta * receiver_dir
            losses[i, j] = get_loss(sender_params, receiver_params) 
    
    A, B = np.meshgrid(alphas, betas)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(A, B, losses.T, cmap='viridis')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    ax.set_zlabel('Loss')
    plt.show()
        
    
def train(model, model_config, sender_optim, receiver_optim, adversary_optim, train_config):
    batch_size = train_config["batch_size"]
    n_steps = train_config["n_steps"]
    eval_steps = train_config["eval_steps"]
    n_samples = train_config["n_samples"]

    sk_dim = model_config["sk_dim"]
    message_dim = model_config["message_dim"]

    rec_losses = []
    adv_losses = []
    losses = []

    pbar = tqdm(range(n_steps))
    for step in pbar:
        sk = random_binary((batch_size, sk_dim))
        message = random_binary((batch_size, message_dim))
        ct, rec_message, adv_message = model(sk, message)
        rec_loss = model.rec_loss(message, rec_message)
        rec_losses.append(rec_loss.item())
        adv_loss = model.adv_loss(message, adv_message)
        adv_losses.append(adv_loss.item())

        loss = rec_loss - adv_loss
    
        sender_optim.zero_grad()
        receiver_optim.zero_grad()
        adversary_optim.zero_grad()
        loss.backward()
        
        adversary_optim.step()
        sender_optim.step()
        receiver_optim.step()
        

        pbar.set_description(f"loss: {loss.item():3f}, rec_loss: {rec_loss.item():3f}, adv_loss: {adv_loss.item():3f}")

        if step % eval_steps == 0:
            print("Samples: \n")
            sk = random_binary((n_samples, sk_dim))
            message = random_binary((n_samples, message_dim))
            ct, rec_message, adv_message = model(sk, message)
            for i in range(n_samples):
                print(f"sk: {sk[i]}")
                print(f"message: {message[i]}")
                print(f"ct: {ct[i]}")
                print(f"rec_message: {rec_message[i]}")
                print(f"adv_message: {adv_message[i]}")
                print()
    
    plt.plot(rec_losses, label="rec_loss")
    plt.plot(adv_losses, label="adv_loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model_config = {
        "sk_dim": 1,
        "message_dim": 1,
        "ct_dim": 1,
        "hidden_dim": 32,
        "depth": 4
    }
    train_config = {
        "batch_size": 512,
        "n_steps": 20000,
        "eval_steps": 1000,
        "n_samples": 5,
    }
    model = Crypto(model_config)
    sk = random_binary((train_config["batch_size"], model_config["sk_dim"]))
    message = random_binary((train_config["batch_size"], model_config["message_dim"]))
    loss_landscape(model, sk, message)


    # sender_optim = torch.optim.SGD(model.sender.parameters(), lr=1e-3)
    # receiver_optim = torch.optim.SGD(model.receiver.parameters(), lr=1e-3)
    # adversary_optim = torch.optim.SGD(model.adversary.parameters(), lr=1e-3, maximize=True)
    # train(model, model_config, sender_optim, receiver_optim, adversary_optim, train_config)





