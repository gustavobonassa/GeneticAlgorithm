import configparser
import matplotlib.pyplot as plt
import numpy as np
import math

parser = configparser.ConfigParser()
parser.read("n_queens.cfg")

def binaryRandom(size):
    return np.random.randint(0, 2, size, dtype="uint8")

def integerPermutation(low, high):
    return np.random.permutation(range(math.floor(low), math.floor(high + 1)))

def integerRandom(size, low, high):
    return np.random.randint(math.floor(low), math.floor(high + 1), size)

def realRandom(size, low, high):
    return np.random.uniform(low, high, size)

def genIndividuo(key):
    dim = int(parser.get("config", "DIM"))
    low = int(parser.get("config", "LOW"))
    high = int(parser.get("config", "HIGH"))

    generator = {
        "BIN": binaryRandom(dim),
        "INT_PERM": integerPermutation(low, high),
        "INT": integerRandom(dim, low, high),
        "REAL": realRandom(dim, low, high),
    }
    return generator[key]

def pmx(parent1, parent2):
    size = min(len(parent1[0]), len(parent2[0]))

    # inicia com 0 dois arrays do mesmo tamanho
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        p1[parent1[0][i] - 1] = i
        p2[parent2[0][i] - 1] = i

    splitPoint1 = np.random.randint(0, size)
    splitPoint2 = np.random.randint(0, size - 1)
    ## ponto 1 precisa ser o menor
    if splitPoint2 >= splitPoint1:
        # evita que o ponto 1 seja igual ao 2
        splitPoint2 += 1
    else:
        # inverte os pontos
        splitPoint1, splitPoint2 = splitPoint2, splitPoint1

    for i in range(splitPoint1, splitPoint2):
        parent1Sol = parent1[0][i]
        parent2Sol = parent2[0][i]

        parent1[0][i], parent1[0][p1[parent2Sol - 1]] = parent2Sol, parent1Sol
        parent2[0][i], parent2[0][p2[parent1Sol - 1]] = parent1Sol, parent2Sol

        p1[parent1Sol - 1], p1[parent2Sol - 1] = p1[parent2Sol - 1], p1[parent1Sol - 1]
        p2[parent1Sol - 1], p2[parent2Sol - 1] = p2[parent2Sol - 1], p2[parent1Sol - 1]

    return (parent1, parent2)

def getPopulation():
    popMax = int(parser.get("config", "POP"))
    key = parser.get("config", "COD")
    pop = []
    ## gera pop
    for i in range(popMax):
        result = genIndividuo(key)
        fitnessVal = None
        pop.append((result, fitnessVal))
    return pop

def fitness(individuo):
    individuoSol = individuo[0]
    dim = len(individuoSol)
    max_fit = dim * (dim - 1)
    fit = max_fit
    for i in range(len(individuoSol) - 1):
        for j in range(i + 1, len(individuoSol)):
            distanceY = abs(individuoSol[i] - individuoSol[j])
            distanceX = abs(i - j)
            fit -= (distanceY == distanceX)
    return (individuoSol, fit / max_fit)

def plotChart(result):
    plt.plot(result)
    plt.show()

def elitismo(population):
    best = population[0]
    for i in range(len(population)):
        if (population[i][1] > best[1]):
            best = population[i]
    return best

def crossover(population, pc):
    change = np.random.rand(len(population))
    for i in range(0, len(change) - 1, 2):
        if change[i] <= pc:
            population[i], population[i + 1] = pmx(
                population[i], population[i + 1]
            )

    return population

def select(population):
    return population

if __name__ == "__main__":
    gen = int(parser.get("config", "GEN"))
    pc = float(parser.get("config", "PC"))
    el = bool(int(parser.get("config", "EL")))

    allElite = []
    best = []

    elite = (None, 0)
    pop = getPopulation()
    for j in range(gen):
        pop = [fitness(i) for i in pop]
        selected = select(pop)
        pop = crossover(selected, pc)

        if el:
            newElite = elitismo(pop)
            if (newElite[1] >= elite[1]):
                elite = newElite
                # bestAllElite.append(newElite[1])
            best.append(newElite[1])

            if (elite[1] > 0):
                pop.pop()
                pop.append(elite)
                

    result = [i[1] for i in pop]
    print(result)
    # plotChart(bestAllElite)
    plotChart(best)
    plotChart(result)