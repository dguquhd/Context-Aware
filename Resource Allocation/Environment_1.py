import numpy as np
import time
import random

from matplotlib import pyplot as plt

import Environment_marl

# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2

n_veh = 4
n_neighbor = 1
n_RB = n_veh




env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game(n_veh)


V2V_SINR_tar = env.V2V_SINR_tar
p_i = env.P
p_0 = 10 ** (env.P_min / 10)
a_step = 0.01
sum_Ps = []
N = []
U_i = []
sum_Us = []
n_episode = 100
n = 0


# ******开始迭代******
def compute_opt_power(self, alpha_mk, alpha_kk, alpha_mB, alpha_kB):
    numK = len(alpha_mk)
    alpha_mk = np.reshape(alpha_mk, (numK, 1))
    phi = -self.gammaProp * np.transpose(alpha_kk)
    for i in range(numK):
        phi[i, i] = alpha_kk[i, i]
    phi_inv = np.linalg.inv(phi)
    num = (self.Pd_max - self.gammaProp * self.sig2 * np.sum(phi_inv, 1))
    den = (self.gammaProp * np.dot(phi_inv, alpha_mk))
    num = np.reshape(num, (numK, 1))
    Pc_cand = num / den
    Pc_opt = min(np.append(Pc_cand, self.Pc_max))
    if Pc_opt <= 0:
        capacity = 0
        Pd_opt = 0
        return capacity, Pc_opt, Pd_opt
    Pd_opt = np.dot(phi_inv, self.gammaProp * (Pc_opt * alpha_mk + self.sig2))
    if np.sum(Pd_opt <= 0) > 1:
        capacity = -1
        return capacity, Pc_opt, Pd_opt
    signal = Pc_opt * alpha_mB
    interference = np.dot(Pd_opt.T, alpha_kB)
    capacity = np.log2(1 + signal / (self.sig2 + interference))
    return capacity, np.squeeze(Pd_opt)
tol = 1e-5
for t in range(n_episode):
    U_i_t0 = 0
    for i in range(len(env.vehicles)):
        U_i_t0 += env.compute_utility(i)  # 每次迭代获得总的效用
    P_list = []
    for i in range(len(env.vehicles)):
        Pi_t0 = env.vehicles[i].P
        P_list.append(Pi_t0)
    loss = []
    for i in range(len(env.vehicles)):
        c = env.c
        alpha_k = 1
        g_k = env.g_k
        I_k = env.V2V_Interference_all
        U = np.log2((P_list[i] * g_k) / I_k - V2V_SINR_tar)
        opt_P = P_list[i] + a_step * ((1 / (np.divide(P_list[i] * g_k, I_k) -V2V_SINR_tar)) * (g_k / I_k) - c * alpha_k)
        if opt_P < p_0:
            Pi_t1 = p_0
        if opt_P >= 10 ** (env.P_max / 10):
            Pi_t1 = 10 ** (env.P_max / 10)
        loss.append(Pi_t1)
        if (P_list[i]*g_k)/I_k <= V2V_SINR_tar:
            U = 0
        if abs(p_i - p_0) < tol:
            opt_P = p_i
    sum_P = 0
    for i in range(len(env.vehicles)):
        env.vehicles[i].P = loss[i]
        sum_P += env.vehicles[i].P
            # if abs(LOSSs[i] - Rs[i]) != 0:
            #     continue
            # cnt += 1
    sum_Ps.append(sum_P)
    env.compute_utility()
    U_i_t1 = 0
    for i in range(len(env.vehicles)):
        U_i_t1 += env.compute_Ui(i)
    print(U_i_t0, U_i_t1)
    sum_Us.append(U_i_t1)
    action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
    F = np.zeros([n_RB, n_veh])
    RB_select = []
    if p_i != opt_P:
        continue
    else:
        for j in range(n_RB):
            for i in range(n_veh):
                for k in range(i+1, n_veh):
                    j = np.argmax(env.compute_utility(p_i))
                    F[j, i] = 1
                # for j in range(n_neighbor):



        n += 1
        tag = 0
        if U_i_t1 < U_i_t0:
            for i in range(len(env.vehicles)):
                env.vehicles[i].c -= 0.005
            tag = 1
            print(tag, n)
            # break
        else:
            for i in range(len(env.vehicles)):
                env.vehicles[i].c += 0.005


fig, ax = plt.subplots()
ax.plot(N, sum_Ps, 'b<:')  # 曲线图，scatter是点状图
ax.legend(["power converge"], loc=4, fontsize='large', facecolor='white', edgecolor='black')  # 图例
ax.set_title("", fontsize=24)
ax.set_xlabel("update number", fontsize=14)
# fig.autofmt_xdate()#绘制倾斜的x轴坐标
ax.set_ylabel("power", fontsize=14)


fig, ax = plt.subplots()
ax.plot(N, U_i, 'b<:')
ax.legend(["Ui converge"], loc=4, fontsize='large', facecolor='white', edgecolor='black')  # 图例
ax.set_title("", fontsize=24)
ax.set_xlabel("update number", fontsize=14)
# fig.autofmt_xdate()#绘制倾斜的x轴坐标
ax.set_ylabel("Ui", fontsize=14)
plt.show()












