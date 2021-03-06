import configparser
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from functools import reduce

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
    size = min(len(parent1), len(parent2))

    # inicia com 0 dois arrays do mesmo tamanho
    p1, p2 = [0] * size, [0] * size

    for i in range(size):
        p1[parent1[i] - 1] = i
        p2[parent2[i] - 1] = i

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
        parent1Sol = parent1[i]
        parent2Sol = parent2[i]

        parent1[i], parent1[p1[parent2Sol - 1]] = parent2Sol, parent1Sol
        parent2[i], parent2[p2[parent1Sol - 1]] = parent1Sol, parent2Sol

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
        pop.append(result)
    return pop

def fitness(individuo):
    # print(individuo)
    dim = len(individuo)
    max_fit = dim * (dim - 1)
    fit = max_fit
    for i in range(len(individuo) - 1):
        for j in range(i + 1, len(individuo)):
            distanceY = abs(individuo[i] - individuo[j])
            distanceX = abs(i - j)
            fit -= 2 * (distanceY == distanceX)
    return (individuo, fit / max_fit)


def plotChart(result, media):
    plt.plot(result, label="Melhor")
    plt.plot(media, label="M??dia")
    plt.legend()
    plt.show()

def elitismo(population):
    best = population[0]
    for i in range(len(population)):
        if (population[i][1] > best[1]):
            best = population[i]
    return best[0].copy(),best[1]

def crossover(population, pc):
    change = np.random.rand(len(population))
    for i in range(0, len(change) - 1, 2):
        if change[i] <= pc:
            population[i], population[i + 1] = pmx(
                population[i], population[i + 1]
            )

    return population

def stochasticTournament(population, k: int = 2, kp: int = 1):
    def fight(subpop):
        lucky_number = random.random()
        if kp >= lucky_number:
            return subpop[1][0]
        return subpop[0][0]

    weights = [1 for i in population]
    sortPop = sorted(
            random.choices(population, k=k, weights=weights), key=lambda x: x[1]
        )[0 :: k - 1]

    newPop = [
            fight(sortPop)
            for i in population
        ]
    return newPop

def fitness_proportionate_selection(population):
    selected = []
    removed_fitness = None
    sum_fit = float(sum([c[1] for c in population]))

    for _ in range(len(population)):
        lucky_number = random.random()
        prev_probability = 0.0

        if removed_fitness is not None:
            population[index] = (population[index][0],removed_fitness)
            sum_fit += removed_fitness
            removed_fitness = None

        for i, c in enumerate(population):
            if (prev_probability + (c[1] / sum_fit)) >= lucky_number:
                selected.append(c[0])
                sum_fit -= c[1]
                removed_fitness = c[1]
                index = i
                c = (c[0],0)
                break
            prev_probability += c[1] / sum_fit

    return selected

def swap(chromosome):
    point_1 = np.random.randint(len(chromosome))
    point_2 = np.random.randint(len(chromosome))
    while point_1 == point_2:
        point_2 = np.random.randint(len(chromosome))
    chromosome[point_1], chromosome[point_2] = chromosome[point_2], chromosome[point_1]

    return chromosome

def select(population):
    return stochasticTournament(population)

def mutate(population):
    def willMutate(individual):
        rnd = np.random.rand()
        return (
            swap(individual)
            if rnd <= float(parser.get("config", "PM"))
            else individual
        )

    return list(map(willMutate, population))

if __name__ == "__main__":
    gen = int(parser.get("config", "GEN"))
    run = int(parser.get("config", "RUN"))
    pc = float(parser.get("config", "PC"))
    el = bool(int(parser.get("config", "EL")))

    bestOfAll = []
    mediaAll = []
    bestSolution = ([], 0)
    for r in range(run):

        bestOfGen = []
        media = []
        
        pop = getPopulation()
        for j in range(gen):
            popFit = [fitness(i) for i in pop]
            avg = reduce(lambda a, b: a + b[1], popFit, 0) / len(popFit)
            media.append(avg)

            elite = elitismo(popFit)
            if elite[1] > bestSolution[1]:
                bestSolution = elite
            bestOfGen.append(elite[1])

            selected = select(popFit)
            cross = crossover(selected, pc)
            pop = mutate(cross)

            if el:
                # adiciona o melhor de volta a populacao
                pop.pop()
                pop.append(elite[0])
                # print(pop)
                # print(selected)
        mediaAll.append(media)
        bestOfAll.append(bestOfGen)

    avgAll = []
    avgAllBest = []
    for j in range(gen):
        avv = 0
        for r in mediaAll:
            avv += r[j]
        md = avv / len(mediaAll)
        avgAll.append(md)

        avv2 = 0
        for r in bestOfAll:
            avv2 += r[j]
        avgAllBest.append(avv2 / len(bestOfAll))
    print("Melhor solucao: ", bestSolution)
    plotChart(avgAllBest, avgAll)