import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess data
digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Simple MLP model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 30)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# Evaluation on clean test set
model.eval()
with torch.no_grad():
    clean_preds = model(X_test).argmax(1)
clean_acc = accuracy_score(y_test.numpy(), clean_preds.numpy())
print(f"\n✅ Clean Accuracy: {clean_acc:.4f}")

# FGSM attack
def fgsm_attack(model, x, y, epsilon=0.5):
    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    perturb = epsilon * x.grad.sign()
    return (x + perturb).detach()



# PGD attack
def pgd_attack(model, x, y, epsilon=0.5, alpha=0.02, steps=40):
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, -3, 3)  # match input range

    for _ in range(steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = criterion(output, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            x_adv = torch.clamp(x_adv, -3, 3)
    return x_adv.detach()


# Run attacks
x_fgsm = fgsm_attack(model, X_test, y_test)
x_pgd = pgd_attack(model, X_test, y_test)

# Predictions
pred_fgsm = model(x_fgsm).argmax(1)
pred_pgd = model(x_pgd).argmax(1)

# Accuracy
acc_fgsm = accuracy_score(y_test.numpy(), pred_fgsm.numpy())
acc_pgd = accuracy_score(y_test.numpy(), pred_pgd.numpy())
print(f"FGSM Accuracy: {acc_fgsm:.4f}")
print(f"PGD Accuracy:  {acc_pgd:.4f}")

# Visualization of first 5 attack results
def show_examples(n=5, save_path="attack_visualization.png"):
    fig, axs = plt.subplots(n, 3, figsize=(9, 2 * n))
    for i in range(n):
        for j, (img, title, pred) in enumerate([
            (X_test[i], f"Clean\nTrue: {y_test[i].item()}\nPred: {clean_preds[i].item()}", clean_preds[i]),
            (x_fgsm[i], f"FGSM\nPred: {pred_fgsm[i].item()}", pred_fgsm[i]),
            (x_pgd[i], f"PGD\nPred: {pred_pgd[i].item()}", pred_pgd[i]),
        ]):
            axs[i, j].imshow(img.detach().numpy().reshape(8, 8), cmap="gray")
            axs[i, j].set_title(title)
            axs[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nAttack visualization saved to: {save_path}")

show_examples(n=5, save_path="attacks_demo.png")

