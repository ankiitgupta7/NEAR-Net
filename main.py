import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
from tqdm import tqdm

# --- Configurable Parameters ---
POP_SIZE = 100       # Population size
N_GEN = 1000          # Number of generations
HIDDEN_SIZE = 30    # Hidden layer size

# --- Load and prepare dataset ---
digits = load_digits()
X = digits.data / 16.0  # Normalize
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_size = X.shape[1]
output_size = 10
IND_SIZE = input_size * HIDDEN_SIZE + HIDDEN_SIZE * output_size + output_size

# --- Genome → Weights ---
def genome_to_weights(genome):
    i = 0
    w1 = np.array(genome[i:i + input_size * HIDDEN_SIZE]).reshape((input_size, HIDDEN_SIZE))
    i += input_size * HIDDEN_SIZE
    w2 = np.array(genome[i:i + HIDDEN_SIZE * output_size]).reshape((HIDDEN_SIZE, output_size))
    i += HIDDEN_SIZE * output_size
    b2 = np.array(genome[i:i + output_size])
    return w1, w2, b2

# --- Evaluation function (accuracy on training set) ---
def evaluate(individual):
    w1, w2, b2 = genome_to_weights(individual)
    hidden = np.tanh(np.dot(X_train, w1))
    output = np.dot(hidden, w2) + b2
    preds = np.argmax(output, axis=1)
    acc = accuracy_score(y_train, preds)
    return (acc,)

# --- DEAP Setup ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Initialize population and Hall of Fame ---
population = toolbox.population(n=POP_SIZE)
hof = tools.HallOfFame(1)

# --- Evolution loop with tqdm ---
progress_bar = tqdm(range(N_GEN), desc="Evolving", unit="gen")
for gen in progress_bar:
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    hof.update(population)
    best_fit = max(ind.fitness.values[0] for ind in population)
    progress_bar.set_postfix(best_accuracy=f"{best_fit:.4f}")

# --- Final evaluation on test set ---
best = hof[0]
w1, w2, b2 = genome_to_weights(best)
hidden = np.tanh(np.dot(X_test, w1))
output = np.dot(hidden, w2) + b2
preds = np.argmax(output, axis=1)
test_acc = accuracy_score(y_test, preds)

print(f"\nFinal test accuracy: {test_acc:.4f}")
