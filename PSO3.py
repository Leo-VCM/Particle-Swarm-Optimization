# 引入库
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

# 定义常数
DIMENSIONS = 2              # Number of dimensions
GLOBAL_BEST = 0             # Global Best of Cost function
B_LO = -5                   # Lower boundary of search space
B_HI = 5                    # Upper boundary of search space

POPULATION = 20             # Number of particles in the swarm
V_MAX = 0.1                 # Maximum velocity value
PERSONAL_C = 2.0            # Personal coefficient factor
SOCIAL_C = 2.0              # Social coefficient factor
CONVERGENCE = 0.001         # Convergence value
MAX_ITER = 100              # Maximum number of iterations

# 定义Particle类
class Particle():
    def __init__(self, x, y, z, velocity):
        self.pos = [x, y]         # 粒子的位置
        self.pos_z = z            # 位置的目标函数值
        self.velocity = velocity  # 粒子的速度
        self.best_pos = self.pos.copy()  # 粒子最佳位置的初始值
        self.best_pos_z = z  # Initialize best_pos_z with the current cost function value

# 定义Swarm类
class Swarm():
    def __init__(self, pop, v_max):
        self.particles = []             # 粒子群
        self.best_pos = None            # 群体中最佳粒子的位置
        self.best_pos_z = math.inf      # 群体中最佳粒子的目标值

        for _ in range(pop):
            x = np.random.uniform(B_LO, B_HI)
            y = np.random.uniform(B_LO, B_HI)
            z = cost_function(x, y)
            velocity = np.random.rand(2) * v_max
            particle = Particle(x, y, z, velocity)
            self.particles.append(particle)
            if particle.pos_z < self.best_pos_z:
                self.best_pos = particle.pos.copy()
                particle.best_pos_z = particle.pos_z  # Update best_pos_z with the new cost function value

# 定义目标函数
def cost_function(x, y, a=20, b=0.2, c=2*math.pi):
    term_1 = np.exp((-b * np.sqrt(0.5 * (x ** 2 + y ** 2))))
    term_2 = np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return -1 * a * term_1 - term_2 + a + np.exp(1)

# 粒子群优化算法
def particle_swarm_optimization():
    # 初始化绘图变量
    x = np.linspace(B_LO, B_HI, 50)
    y = np.linspace(B_LO, B_HI, 50)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure("Particle Swarm Optimization")

    # 初始化绘图变量
    Z = np.array([cost_function(i, j) for i, j in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)

    # 绘制目标函数的3D曲面
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    # 初始化粒子群
    swarm = Swarm(POPULATION, V_MAX)

    # 主循环
    for iteration in range(MAX_ITER):
        for particle in swarm.particles:
            if particle.pos_z < particle.best_pos_z:
                particle.best_pos = particle.pos.copy()
                particle.best_pos_z = particle.pos_z
            if particle.pos_z < swarm.best_pos_z:
                swarm.best_pos = particle.pos.copy()
                swarm.best_pos_z = particle.pos_z

            # 更新粒子速度和位置
            r1 = np.random.rand()
            r2 = np.random.rand()
            cognitive_velocity = PERSONAL_C * r1 * (np.array(particle.best_pos) - np.array(particle.pos))
            social_velocity = SOCIAL_C * r2 * (np.array(swarm.best_pos) - np.array(particle.pos))
            particle.velocity += cognitive_velocity + social_velocity
            particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)
            particle.pos += particle.velocity

            # 重新计算目标函数值
            particle.pos_z = cost_function(particle.pos[0], particle.pos[1])

        # 可视化粒子的运动
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
        for particle in swarm.particles:
            ax.scatter(particle.pos[0], particle.pos[1], particle.pos_z, color='red')
        plt.pause(0.1)

    # 显示最终结果
    plt.show()

    # 显示最佳解
    print(f"最佳解: {swarm.best_pos}")
    print(f"最佳目标函数值: {swarm.best_pos_z}")

# 主函数调用
if __name__ == "__main__":
    particle_swarm_optimization()
