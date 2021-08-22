import configparser
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from functools import reduce

parser = configparser.ConfigParser()
parser.read("n_queens_valorado.cfg")

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

def show_table(chromosome, profit_array, max_fit_profit) -> str:
    size = len(chromosome)
    board = np.full((size, size), ".")

    for index, item in enumerate(chromosome):
        board[index][item - 1] = "x"

    out = ""
    for row in board:
        out += "|"
        for tile in row:
            out += tile
        out += " |\n"

    col, profit = 0, 0
    for i in range(len(chromosome) - 1):
        profit += profit_array[(i + 1) * (chromosome[i] - 1)]
        for j in range(i + 1, len(chromosome)):
            col += 2 * (abs(chromosome[i] - chromosome[j]) == abs(i - j))

    out += f"\ncollisions: {col}\n"
    out += "profit: %f/%f\n" % (profit, max_fit_profit)

    return out

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

def fitness(individo, values, max_fit_profit):
    dim = len(individo)
    fit = dim * (dim - 1)
    max_fit = fit * 0.8 + max_fit_profit * 0.2

    profit: float = 0
    for i in range(len(individo) - 1):
        profit += values[(i + 1) * (individo[i] - 1)]
        for j in range(i + 1, len(individo)):
            fit -= 2 * (abs(individo[i] - individo[j]) == abs(i - j))

    return individo, (fit * 0.8 + profit * 0.2) / max_fit


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

def fitnessProportionateSelection(population):
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
    dim = int(parser.get("config", "DIM"))

    profit_array = np.array(list(map(float, range(1, dim ** 2 + 1))))
    op, po = math.sqrt, math.log10
    for i in range(len(profit_array)):
        profit_array[i] = op(profit_array[i])
        if not (i + 1) % dim:
            po, op = op, po
            continue

    max_fit_profit = sum(sorted(profit_array)[-1 : -dim - 1 : -1])
    bestOfAll = []
    mediaAll = []
    bestSolution = ([], 0)
    for r in range(run):

        bestOfGen = []
        media = []
        
        pop = getPopulation()
        for j in range(gen):
            popFit = [fitness(i, profit_array, max_fit_profit) for i in pop]
            avg = reduce(lambda a, b: a + b[1], popFit, 0) / len(popFit)
            media.append(avg)
            # print(popFit)

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
    # print(show_table(bestSolution[0], profit_array, max_fit_profit))
    plotChart(avgAllBest, avgAll)