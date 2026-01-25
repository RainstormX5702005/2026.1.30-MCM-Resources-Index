from typing import Self
import numpy as np

# 使用 GA 求解函数 f(x) = Ackley function 的最小值


def ackley(x):
    """f(x) = -a * exp(-b * sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(c*x_i))) + a + exp(1)"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


class GeneticAlgorithmGenerator:
    """遗传算法生成器用于求解 Ackley 函数最小值问题"""

    def __init__(
        self,
        *,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        max_iters: int = 100,
        bounds: np.ndarray = np.array([-20, 20]),
        individuals: np.ndarray = np.zeros((100, 1)),
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_iters = max_iters
        self.bounds = bounds  # (min, max)
        self.individuals = (
            individuals
            if individuals is not np.zeros((population_size, 1))
            else self._initialize_population()
        )

    def _initialize_population(self) -> np.ndarray:
        return np.random.uniform(
            self.bounds[0], self.bounds[1], (self.population_size, 1)
        )

    def _fitness(self, ind: np.ndarray) -> float:
        return ackley(ind)

    def mutation(self, ind: np.ndarray) -> Self:
        """对于连续型变量，使用高斯扰动进行变异，即随机正态分布采样并加到原本的个体上"""
        for i in range(len(ind)):
            if np.random.rand() < self.mutation_rate:
                mval = np.random.normal(0, 1)
                ind[i] += mval
                # 进行必要的边界截断，超越边界就按照边界计算
                ind[i] = np.clip(ind[i], self.bounds[0], self.bounds[1])
        return self

    def crossover(self, ind: np.ndarray) -> Self:
        """`crossover` 方法：用于实现两个个体之间的交叉操作，生成新的个体

        实现 crossover 的方法有很多，这里使用经典的俄罗斯轮盘赌选择法来处理这些变量。
        """

        # 1. 计算每个个体的 fitness FIXME: 这里错误的问题在于我们寻找的是最小值，而目前的逻辑是寻找最大值
        fitness = np.array([self._fitness(individual) for individual in ind])
        fitness = 1 / (fitness + 1e-6)  # 转化为越大越好的问题
        # 2. 进行概率的归一化
        fitness /= fitness.sum()
        # 3. 生成累积概率函数
        cumulative_fitness = np.cumsum(fitness)
        # 4. 通过概率选择个体并交叉，这里选择偶数全分配策略
        new_inds = []
        for i in range(0, len(ind), 2):
            if np.random.rand() < self.crossover_rate:
                # 通过生成的概率来决定进行交配的个体
                idx1 = np.searchsorted(cumulative_fitness, np.random.uniform(0, 1))
                idx2 = np.searchsorted(cumulative_fitness, np.random.uniform(0, 1))
                p1 = ind[idx1]
                p2 = ind[idx2]

                # 使用 linear-gradient crossover 的方法来计算交配后的个体
                alpha = np.random.rand()
                child1 = alpha * p1 + (1 - alpha) * p2
                child2 = (1 - alpha) * p1 + alpha * p2
                new_inds.extend([child1, child2])
            else:
                new_inds.extend([ind[i], ind[i + 1]])

        self.individuals = np.array(new_inds)
        return self

    def ga_result(self) -> tuple[np.ndarray, float]:
        for iter in range(self.max_iters):
            self.crossover(self.individuals).mutation(self.individuals)
        fitness_values = np.array([self._fitness(ind) for ind in self.individuals])
        best_idx = np.argmin(fitness_values)
        best_individual = self.individuals[best_idx]
        best_fitness = fitness_values[best_idx]
        return best_individual, best_fitness
