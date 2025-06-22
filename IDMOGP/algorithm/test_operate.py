# import numpy as np
# import matplotlib.pyplot as plt
# def f(x):
#     return np.exp(-1/np.sqrt(np.sqrt(x)))
# a = []
# for i in range(1, 1000):
#     a.append(f(i))
# plt.scatter(range(1, 1000), a)
# plt.scatter(range(1, 1000), range(1, 1000))
# plt.show()
def fast_non_dominated_sort(population):
    # 初始化非支配级别和支配该级别的解的集合
    n = len(population)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts = []

    for i in range(n):
        for j in range(i + 1, n):
            is_dominated = all(population[i][k] <= population[j][k] for k in range(len(population[i])))
            if is_dominated:
                domination_count[j] += 1
                dominated_solutions[i].append(j)
            else:
                domination_count[i] += 1
                dominated_solutions[j].append(i)

    current_front = [i for i in range(n) if domination_count[i] == 0]
    fronts.append(current_front)
    front_idx = 0

    while current_front:
        next_front = []
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front_idx += 1
        current_front = next_front

    return fronts

# 示例用法
if __name__ == "__main__":
    population = [(2, 3), (1, 4), (3, 2), (4, 1), (2, 4), (3, 1)]  # 示例解决方案，每个解包含两个目标值

    fronts = fast_non_dominated_sort(population)

    for i, front in enumerate(fronts):
        print(f"Front {i + 1}: {front}")
