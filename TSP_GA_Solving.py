import numpy as np
import random
from tqdm import tqdm

num_of_cities = 30 # 노드수
min_weight_range = 1 # 가중치 최소값
max_weight_range = 31 # 가중치 최대값
size_of_pop = 300 # 한 개체군의 개체 수
normalize_fitness = False # 적합도 점수 정규화 여부 결정
if normalize_fitness is True: # 반복 멈출 적합도 점수 최소치 결정
    fitness_threshold = 0.8
else:
    fitness_threshold = 750
show_info_num = 500 # 출력 빈도 결정(반비례)
mutation_freq = 50 # 돌연변이 출현 빈도 결정(반비례)

graph = [[0] * num_of_cities for _ in range(num_of_cities)]

for i in range(num_of_cities):
    for j in range(num_of_cities):
        if i == j:
            continue
        graph[i][j] = random.randrange(min_weight_range, max_weight_range)
# 인접 행렬로 그래프 생성
for i in range(len(graph)):
    print(graph[i])

population = []
chromosome = []

for i in range(size_of_pop):
    population.append(random.sample(range(min_weight_range-1, max_weight_range-1), num_of_cities))


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

def best_chroms_selection(population, num = 2):
    _, fitness_of_chroms = cal_fitness_of_population(population, normalize_fitness)

    np_score = np.array(fitness_of_chroms)
    inds = np.argsort(np_score)[::-num]
    return inds

def roulette_wheel_selection(population, num = 2):
    _, fitness_of_chroms = cal_fitness_of_population(population, normalize_fitness)
    sum_of_scores = sum(fitness_of_chroms)
    num_rand = random.sample(range(0, sum_of_scores), num)
    accumulated_score = [0] * len(population)
    accumulated_score[0] = fitness_of_chroms[0]
    for i in range(len(accumulated_score)-1):
        accumulated_score[i + 1] = accumulated_score[i] + fitness_of_chroms[i+1]
    #랜덤수 하나하나보다 큰 누적 적합도 갖는 인덱스 뽑아내서 모아
    indices_selected = []
    for i in range(len(num_rand)):
        indices_selected.append(sequential_search(accumulated_score, num_rand[i]))
    return indices_selected

def sequential_search(arr, target):
    for i in range(len(arr)):
        if arr[i] >= target:
            return i
    return None

def uniform_crossover(p1, p2, crossover_rate):
    c1 = p1.copy()
    c2 = p2.copy()
    m = np.random.rand(30)
    m = list(m)
    for i in range(len(m)):
        if m[i] > crossover_rate:
            temp = c1[i]
            c1[i] = c2[i]
            c2[i] = temp
    return c1, c2

def cyclic_crossover(p1, p2):
    c1 = [-1] * 30
    c2 = [-1] * 30
    c1[0] = p1[0]
    c2[0] = p2[0]

    index = 0
    while True:
        value = p2[index]
        index = p1.index(value)
        if index == 0:
            break
        c1[index] = p1[index]

    index = 0
    while True:
        value = p1[index]
        index = p2.index(value)
        if index == 0:
            break
        c2[index] = p2[index]

    print()
    print(c1, c2)

    while True:
        if -1 not in c1:
            break
        index_of_none = c1.index(-1)
        c1[index_of_none] = p2[index_of_none]

    while True:
        if -1 not in c2:
            break
        index_of_none = c2.index(-1)
        c2[index_of_none] = p1[index_of_none]

    print(c1, c2)
    return c1, c2

def cyclic_crossover_ver_numpy(p1, p2):
    c1 = np.array([-1] * 30)
    c2 = np.array([-1] * 30)
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

def mutation_oper(chrom, mutation_rate):
    for i in range(mutation_rate):
        while True:
            i1 = random.randint(0, 29)
            i2 = random.randint(0, 29)
            if i1 != i2:
                break
        temp = chrom[i1]
        chrom[i1] = chrom[i2]
        chrom[i2] = temp


def cal_sum_of_weights(chrom):
    sum_of_weights = 0
    for i in range(num_of_cities - 1):
        sum_of_weights += graph[chrom[i]][chrom[i+1]]
    return sum_of_weights

population, score = sort_pop_by_score(population)

fitness_of_pop, fitness_of_chroms = cal_fitness_of_population(population, normalize_fitness)

for count in tqdm(range(100000)):
    population, fitness_of_chroms = sort_pop_by_score(population)

    sel_indices = roulette_wheel_selection(population, 10)
    #sel_indices = best_chroms_selection(population, 10)
    temp_child = []
    for i in range(len(sel_indices) - 1):
        c1, c2 = cyclic_crossover_ver_numpy(population[sel_indices[i]], population[sel_indices[i+1]])
        if count % mutation_freq == 0:
            mutation_oper(c1, 2)
            mutation_oper(c2, 2)

        temp_child.append(c1)
        temp_child.append(c2)

    for i in range(len(temp_child)):
        population.pop()
    for i in range(len(temp_child)):
        population.append(temp_child[i])

    fitness_of_pop, fitness_of_chroms = cal_fitness_of_population(population, normalize_fitness)

    if count % show_info_num == 0:
        print()
        print(fitness_of_chroms[0])
        print(fitness_of_pop)
        print(cal_sum_of_weights(population[0]))

    count += 1

    if fitness_of_chroms[0] >= fitness_threshold: break

population, fitness_of_chroms = sort_pop_by_score(population)
print(fitness_of_pop)
print(fitness_of_chroms)
print(population[0])
print(cal_sum_of_weights(population[0]))
#stop rule은 최대 세대수 1개(필)와 목표 적합도 도달, 동일값 반복 도출, 지정 시간 경과 중 택 1하여 총 2개 구현
#교배 연산자 ordered crossover, cyclic crossover 두 개도 추가