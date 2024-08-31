import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False


DNA_SIZE = 9
POP_SIZE = 10
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.01
N_GENERATIONS = 1000
X_BOUND = [-5,5]
X_V = []
for j in range(30):
    bit = (X_BOUND[1] - X_BOUND[0])/1e-2
    if (2**j < bit and 2**(j+1) > bit):
        DNA_SIZE = j+1
        break
print("DNA_SIZE:",DNA_SIZE)
        

def F(x):
    return -x**2 - 4*x + 1  

def get_fitness(pop):
    x = translateDNA(pop)
    pred = F(x)
    # return pred
    return pred - np.min(pred)+1e-3  # 求最大值时的适应度
    # fitness = np.max(pred) - pred + 1e-6  # 求最小值时的适应度，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)]
    # return fitness


def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop.copy()  
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    return x


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)
    new_pop[0] = pop[max_fitness_index] # 保留精英

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制位反转


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("x:",x[max_fitness_index])
    print(F(x[max_fitness_index]))


if __name__ == "__main__":
    plt.figure()
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))  # matrix (POP_SIZE, DNA_SIZE)

    for i in range(N_GENERATIONS):  # 迭代N代
        x = translateDNA(pop)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        fitness = get_fitness(pop)
        max_fitness_index = np.argmax(fitness)
        x = translateDNA(pop)
        X_V.append(F(x[max_fitness_index]))
        pop = select(pop, fitness)  # 选择生成新的种群
    # X_V
    print_info(pop)
    plt.plot(list(range(0,N_GENERATIONS)),X_V,color='b')
    plt.xlabel('代数')
    plt.ylabel('最大适应度')
    plt.title('遗传算法适应度曲线')
    plt.show()