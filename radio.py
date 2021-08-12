import configparser
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from functools import reduce

parser = configparser.ConfigParser()
parser.read("radio.cfg")

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

def binaryToInt(individual):
    out = 0
    for bit in individual:
        out = (out << 1) | bit

    return out

def normalization(ind):
    xrl = binaryToInt(ind[: len(ind) // 2])
    xrs = binaryToInt(ind[len(ind) // 2 :])
    # rs [0-24] -> 5 bits
    # rl [0-16] -> 5 bits
    rl = math.floor(0 + (16 / 31) * xrl)
    rs = math.floor(0 + (24 / 31) * xrs)

    return rl, rs

def fitness(ind):
    rl, rs = normalization(ind)

    r = -1
    FOn = (30 * rs + 40 * rl) / 1360
    Hn = max(0, (rs + 2 * rl - 40)) / 16
    fit = FOn + r * Hn

    return ind, fit

def plotChart(result, media):
    plt.plot(result, label="Melhor")
    plt.plot(media, label="Média")
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
            population[i], population[i + 1] = two_points(
                population[i], population[i + 1]
            )

    return population

def stochastic_tournament(population, k: int = 2, kp: int = 1):
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

def bit_flip(chromosome):
    choose = np.random.rand(len(chromosome))

    for i in range(len(choose)):
        if choose[i] < 0.5:
            chromosome[i] = 1 if chromosome[i] == 0 else 0

    return chromosome


def select(population):
    return fitness_proportionate_selection(population)

def one_point(
    parent_1, parent_2, index = -1
):
    if index == -1:
        index = int(np.random.uniform(low=1, high=len(parent_1) - 1))
    return np.array(
        [
            np.hstack([parent_1[:index], parent_2[index:]]),
            np.hstack([parent_2[:index], parent_1[index:]]),
        ]
    )


def two_points(parent_1, parent_2):
    crom_size = len(parent_1) - 1

    index_1 = int(np.random.uniform(low=1, high=crom_size - 2))
    index_2 = int(np.random.uniform(low=index_1, high=crom_size))

    child_1, child_2 = one_point(parent_1, parent_2, index_1)

    return one_point(child_1, child_2, index_2)


def uniform(parent_1, parent_2):
    choose = np.random.rand(len(parent_1))

    for i in range(len(choose)):
        if choose[i] < 0.5:
            temp = parent_1[i]
            parent_1[i] = parent_2[i]
            parent_2[i] = temp
    return [parent_1, parent_2]

def mutate(population):
    def will_mutate(individual):
        rnd = np.random.rand()
        return (
            bit_flip(individual)
            if rnd <= float(parser.get("config", "PM"))
            else individual
        )

    return list(map(will_mutate, population))

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