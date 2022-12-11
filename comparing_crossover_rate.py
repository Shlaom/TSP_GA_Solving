import numpy as np
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

#TSP, Hyperparameters
num_of_cities = 100 # 노드수
min_weight_range = 1 # 가중치 최소값
max_weight_range = num_of_cities + min_weight_range # 가중치 최대값
size_of_pop = 80 # 한 개체군의 개체 수
normalize_fitness = True # 적합도 점수 정규화 여부 결정
if normalize_fitness is True: # 반복 멈출 적합도 점수 최소치 결정
    fitness_threshold = 0.90
else:
    fitness_threshold = 0.85 * ((max_weight_range-1) * num_of_cities - min_weight_range * num_of_cities) \
                        + min_weight_range * num_of_cities
GA_iter = 15000
crossover_rate = 15
show_info_num = 500
mutation_rate = 2
num_mutation_exchange = 1

graph = [[0] * num_of_cities for _ in range(num_of_cities)]

# 인접 행렬로 그래프 생성
for i in range(num_of_cities):
    for j in range(num_of_cities):
        if i == j:
            continue
        graph[i][j] = random.randrange(min_weight_range, max_weight_range)

for i in range(len(graph)):
    print(graph[i])

population = []

for i in range(size_of_pop):
    population.append(random.sample(range(min_weight_range-1, max_weight_range-1), num_of_cities))


def make_pop():
    population = []

    for i in range(size_of_pop):
        population.append(random.sample(range(min_weight_range - 1, max_weight_range - 1), num_of_cities))

    return population

def sort_pop_by_score(population):
    fitness_of_pop, fitness_of_chroms = cal_fitness_of_population(population, normalize_fitness)

    sorted_population = np.array(population)
    inds = np.array(fitness_of_chroms)
    inds = np.argsort(inds)[::-1]
    sorted_population = sorted_population[inds]
    sorted_population = list(sorted_population)
    fitness_of_chroms.sort(reverse=True)
    return sorted_population, fitness_of_chroms

def cal_fitness_of_chrom(chrom, normalize=True):
    sum_of_weights = cal_sum_of_weights(chrom)
    max = (max_weight_range - 1) * len(chrom)
    min = (min_weight_range) * len(chrom)
    score = max - sum_of_weights
    if normalize is True:
        normed_score = (score - min) / (max - min)
        return normed_score
    else:
        return score

def cal_fitness_of_population(population, normalize=True):
    fitness_of_chroms = []
    for i in range(len(population)):
        fitness_of_chroms.append(cal_fitness_of_chrom(population[i], normalize))
    unnormed_fitness_of_population = sum(fitness_of_chroms)
    if normalize is True:
        max = len(population)
        min = 0
        normed_fitness_of_population = (unnormed_fitness_of_population - min) / (max - min)
        return normed_fitness_of_population, fitness_of_chroms
    else:
        return unnormed_fitness_of_population, fitness_of_chroms

def cal_sum_of_weights(chrom):
    sum_of_weights = 0
    for i in range(num_of_cities - 1):
        sum_of_weights += graph[chrom[i]][chrom[i+1]]
    return sum_of_weights

def sequential_search(arr, target):
    for i in range(len(arr)):
        if arr[i] >= target:
            return i
    return None


#selection 기법
def best_chroms_selection(population, num = 2):
    _, fitness_of_chroms = cal_fitness_of_population(population, normalize_fitness)

    np_score = np.array(fitness_of_chroms)
    inds = np.argsort(np_score)[::-num]
    return inds

def roulette_wheel_selection(population, num = 2):
    _, fitness_of_chroms = cal_fitness_of_population(population, normalize=False)
    sum_of_scores = sum(fitness_of_chroms)
    num_rand = random.sample(range(0, sum_of_scores), num)
    accumulated_score = [0] * len(population)
    accumulated_score[0] = fitness_of_chroms[0]
    for i in range(len(accumulated_score)-1):
        accumulated_score[i + 1] = accumulated_score[i] + fitness_of_chroms[i+1]

    indices_selected = []
    for i in range(len(num_rand)):
        indices_selected.append(sequential_search(accumulated_score, num_rand[i]))
    return indices_selected


#crossover 기법
def cyclic_crossover_ver_numpy(p1, p2):
    c1 = np.array([-1] * len(p1))
    c2 = np.array([-1] * len(p1))
    c1[0] = p1[0]
    c2[0] = p2[0]

    index = 0
    passed_index = []
    while True:
        value = p2[index]
        index = np.where(p1 == value)[0][0]
        if index in passed_index:
            break
        c1[index] = p1[index]
        passed_index.append(index)

    index = 0
    passed_index = []
    while True:
        value = p1[index]
        index = np.where(p2 == value)[0][0]
        if index in passed_index:
            break
        c2[index] = p2[index]
        passed_index.append(index)

    index_of_none = np.where(c1 == -1)
    for i in range(len(index_of_none)):
        c1[index_of_none] = p2[index_of_none]

    index_of_none = np.where(c2 == -1)
    for i in range(len(index_of_none)):
        c2[index_of_none] = p1[index_of_none]

    return c1, c2

def ordered_crossover(p1, p2):
    wall = int(len(p1)/3)
    c1 = np.array([-1] * len(p1))
    c2 = np.array([-1] * len(p2))

    c1[wall:wall * 2] = p1[wall:wall * 2]
    c2[wall:wall * 2] = p2[wall:wall * 2]

    temp_p1 = p1.copy()
    temp_p2 = p2.copy()
    a = p1[wall:wall * 2]
    b = p2[wall:wall * 2]
    indices1, indices2 = [], []
    for i in range(len(a)):
        indices1.append(np.where(temp_p2 == a[i])[0][0])
        indices2.append(np.where(temp_p1 == b[i])[0][0])
    for i in range(len(indices1)):
        temp_p2[indices1[i]] = -1
        temp_p1[indices2[i]] = -1
    while -1 in temp_p2:
        i = np.where(temp_p2 == -1)[0][0]
        temp_p2 = np.delete(temp_p2, i)
    while -1 in temp_p1:
        i = np.where(temp_p1 == -1)[0][0]
        temp_p1 = np.delete(temp_p1, i)

    c1[:wall] = temp_p2[:wall]
    c1[wall * 2:] = temp_p2[wall:]
    c2[:wall] = temp_p1[:wall]
    c2[wall * 2:] = temp_p1[wall:]
    return c1, c2


#돌연변이 연산
def mutation_oper(chrom, mutation_rate):
    for i in range(mutation_rate):
        while True:
            i1 = random.randint(0, num_of_cities -1)
            i2 = random.randint(0, num_of_cities -1)
            if i1 != i2:
                break
        temp = chrom[i1]
        chrom[i1] = chrom[i2]
        chrom[i2] = temp


#GA 실행
crossover_rate = 5
fitness_threshold = 0.9
origin_population = make_pop()

for p in range(9):
    score_list = []
    population = origin_population.copy()
    population = make_pop()
    generation = GA_iter

    start_time = time.time()
    for count in tqdm(range(GA_iter)):
        population, fitness_of_chroms = sort_pop_by_score(population)

        #sel_indices = roulette_wheel_selection(population, crossover_rate)
        sel_indices = best_chroms_selection(population, crossover_rate)
        temp_child = []
        for i in range(len(sel_indices) - 1):
            c1, c2 = cyclic_crossover_ver_numpy(population[sel_indices[i]], population[sel_indices[i+1]])
            #c1, c2 = ordered_crossover(population[sel_indices[i]], population[sel_indices[i + 1]])
            if count % mutation_rate == 0:
                mutation_oper(c1, num_mutation_exchange)
                mutation_oper(c2, num_mutation_exchange)

            temp_child.append(c1)
            temp_child.append(c2)

        for i in range(len(temp_child)):
            population.pop()
        for i in range(len(temp_child)):
            population.append(temp_child[i])

        fitness_of_pop, fitness_of_chroms = cal_fitness_of_population(population, normalize_fitness)

        '''if count % show_info_num == 0:
            print()
            print('Max fitness of chromosomes: ', fitness_of_chroms[0])
            print('Present fitness of population: ', fitness_of_pop)
            print('Sum of weights: ', cal_sum_of_weights(population[0]))
            print("============================================================")'''

        count += 1
        _, fitness_of_chroms = sort_pop_by_score(population)
        score_list.append(fitness_of_chroms[0])

        if fitness_of_chroms[0] >= fitness_threshold:
            generation = count
            break

    end_time = time.time()

    fitness_of_pop, _ = cal_fitness_of_population(population, normalize_fitness)
    population, fitness_of_chroms = sort_pop_by_score(population)
    print()
    print('Fitness of population: ', fitness_of_pop)
    print('Fitnesses of chromosomes: ', fitness_of_chroms)
    print('Max fitness chromosome info: ', population[0])
    print("Max fitness chromosome's sum of weights: ", cal_sum_of_weights(population[0]))
    print('crossover_rate: ', crossover_rate)
    print("==========================================================================")
    plt.subplot(3, 3, p+1)
    plt.plot(score_list)
    plt.ylim([0.5, 0.9])
    plt.xlim([0, 15000])
    elapsed_time = end_time - start_time
    plt.title('Crossover rate: ' + str(crossover_rate) + '  /  ' + str(round(elapsed_time)) + 'sec  /  generation: ' + str(generation))
    plt.xlabel('Num of iteration')
    plt.ylabel('Fitness score')
    plt.grid(True)
    crossover_rate += 2
    time.sleep(1)
plt.suptitle('size of pop: ' + str(size_of_pop) + '   /   Mutation rate: ' + str(mutation_rate) + '   /   Num of mutation exchange: ' + str(num_mutation_exchange))
plt.show()