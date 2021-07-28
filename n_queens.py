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

def elitism(population):
    best = population[0]
    for i in range(len(population)):
        if (population[i][1] > best[1]):
            best = population[i]
    return best

if __name__ == "__main__":
    gen = int(parser.get("config", "GEN"))

    allElite = []
    elite = (None, 0)
    pop = getPopulation()
    for j in range(gen):
        ## mutacao da pop
        # if (elite[1] > 0):
        #     pop.append(elite)
        pop = [fitness(i) for i in pop]
        newElite = elitism(pop)
        # print(newElite)
        # if (newElite[1] > elite[1]):
        #     elite = newElite
        #     allElite.append(newElite[1])

    result = [i[1] for i in pop]

    # plotChart(allElite)
    plotChart(result)