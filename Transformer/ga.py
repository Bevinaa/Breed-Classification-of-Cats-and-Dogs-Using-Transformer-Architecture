import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from deap import base, creator, tools
import random
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------------------
# Set seed for reproducibility
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ------------------------------
# 1. Define Residual MLP Model
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_prob=0.4):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(size, size)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10, dropout_prob=0.4):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim, dropout_prob)
        self.res_block2 = ResidualBlock(hidden_dim, dropout_prob)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.output_layer(x)
        return x

# ------------------------------
# 2. Training function
# ------------------------------
def train_residual_mlp(train_features, train_labels,
                       val_features, val_labels,
                       epochs=5,
                       batch_size=64,
                       learning_rate=0.001,
                       dropout_prob=0.4,
                       weight_decay=1e-4,
                       optimizer_name='adamw',
                       device='cpu'):
    input_dim = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))

    model = ResidualMLP(input_dim, hidden_dim=128, num_classes=num_classes, dropout_prob=dropout_prob)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return val_acc, model

# ------------------------------
# 3. GA Hyperparameter tuning
# ------------------------------
def evaluate_individual(individual):
    lr, dropout, wd = individual

    # Clamp hyperparameters to valid ranges
    lr = max(0.0001, min(0.01, lr))
    dropout = max(0.3, min(0.5, dropout))
    wd = max(1e-6, min(1e-3, wd))

    val_acc, _ = train_residual_mlp(   # unpack here
        train_features, train_labels,
        val_features, val_labels,
        epochs=5,
        learning_rate=lr,
        dropout_prob=dropout,
        weight_decay=wd,
        device=device
    )

    print(f"LR={lr:.5f}, Dropout={dropout:.3f}, WD={wd:.6f} -> Val Acc={val_acc:.4f}")
    return val_acc,

def run_ga_optimization(pop_size=10, ngen=5):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_lr", random.uniform, 0.0001, 0.01)
    toolbox.register("attr_dropout", random.uniform, 0.3, 0.5)
    toolbox.register("attr_wd", random.uniform, 1e-6, 1e-3)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_lr, toolbox.attr_dropout, toolbox.attr_wd), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)

    for gen in range(ngen):
        print(f"Generation {gen+1}")
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Best accuracy so far: {max(fits):.4f}")

    best = tools.selBest(pop, 1)[0]
    print(f"Best individual: {best}, accuracy: {best.fitness.values[0]:.4f}")
    return best

# ------------------------------
# 4. Load your saved data and split
# ------------------------------
train_features_np = np.load('combined_features.npy')  # shape: (num_samples, feature_dim)
train_labels_np = np.load('labels.npy')               # shape: (num_samples, )

train_features_np, val_features_np, train_labels_np, val_labels_np = train_test_split(
    train_features_np, train_labels_np, test_size=0.2, random_state=42, stratify=train_labels_np)

train_features = torch.from_numpy(train_features_np).float()
train_labels = torch.from_numpy(train_labels_np).long()

val_features = torch.from_numpy(val_features_np).float()
val_labels = torch.from_numpy(val_labels_np).long()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------
# 5. Run GA Optimization & final training
# ------------------------------
if __name__ == "__main__":
    best_hyperparams = run_ga_optimization(pop_size=10, ngen=5)

    final_acc, model = train_residual_mlp(
        train_features, train_labels,
        val_features, val_labels,
        epochs=30,
        learning_rate=best_hyperparams[0],
        dropout_prob=best_hyperparams[1],
        weight_decay=best_hyperparams[2],
        device=device
    )
    print(f"\nFinal validation accuracy after full training: {final_acc * 100:.2f}%")

    model_path = "residual_mlp_final.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved final Residual MLP model to {model_path}")

