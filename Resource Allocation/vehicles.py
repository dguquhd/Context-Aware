# 在此环境中均匀生成车辆：
import numpy as np
import random
import math

class Vehicle:
    def __init__(self, start_pos, start_direction, velocity):
        self.position = start_pos
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []
class Environ:
  # 车辆的模拟环境
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height
        self.vehicles = []

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def generate_vehicles(self, num_vehicles):
        # vehicles = []
        start_poss = []
        velocitys = []
        directions = ['u', 'd', 'l', 'r']
        for i in range(num_vehicles):
            ind = np.random.randint(0, len(self.down_lanes))
            start_direction = random.choice(directions)
            if start_direction == 'u':
                start_direction = 'u'
                start_pos = [self.up_lanes[ind], np.random.randint(0, self.height)]
            elif start_direction == 'd':
                start_direction = 'd'
                start_pos = [self.down_lanes[ind], np.random.randint(0, self.height)]
            elif start_direction == 'l':
                start_direction = 'l'
                start_pos = [np.random.randint(0, self.width), self.left_lanes[ind]]
            else:
                start_direction = 'r'
                start_pos = [np.random.randint(0, self.width), self.right_lanes[ind]]
            # 在环境中随机选择一条车道
            # lane = random.choice([self.up_lanes, self.down_lanes, self.left_lanes, self.right_lanes])
            # 在车道上随机选择一个位置作为起始位置
            # if lane = self.up_lanes or self.down_lanes:
            # start_pos = (random.uniform(0, lane[-1]), random.uniform(0, env.height))
            # 在[-pi/2, pi/2]范围内随机选择一个速度方向
            # start_dir = random.uniform(-math.pi / 2, math.pi / 2)
            # 随机生成一个速度大小
            velocity = random.uniform(10, 30)
            # 创建Vehicle对象并添加到列表中
            # vehicle = Vehicle(start_pos, start_direction, velocity)
            self.add_new_vehicles(start_pos, start_direction, velocity)
            start_poss.append(start_pos)
            velocitys.append(velocity)
            # print(num_vehicles)
            # print('位置：', start_pos)
            # # print('速度：', velocity)
            # print('方向：', start_direction)
        return velocitys, start_poss


    # def calculate_distance(self, vehicle1, vehicle2):
    #     # 计算两个车辆之间的距离
    #     x1, y1 = vehicle1.position
    #     x2, y2 = vehicle2.position
    #     distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #     return distance


    def generate_V2I_and_V2V(self, numV2I, numV2V):
        # 从生成的所有车辆中随机选择numV2I个当作V2I
        # print(len(self.vehicles))
        indvehicles = np.random.permutation(len(self.vehicles))
        v2v_senders = indvehicles[:numV2V]
        v2v_receivers = -np.ones(numV2V, dtype=np.int)
        # print(start_poss[1])
        for i in range(numV2V):
            minDist = np.inf
            tmpInd = 0
            for j in range(len(self.vehicles)):
                if j in v2v_senders or j in v2v_receivers:
                    continue
                # print(start_poss[v2v_senders[i]][0])
                # newDist = np.sqrt(((start_poss[v2v_senders[i]] - start_poss[j])**2))
                newDist = np.sqrt((start_poss[v2v_senders[i]][0] - start_poss[j][0])**2 + (start_poss[v2v_senders[i]][1] - start_poss[j][1])**2)
                if newDist < minDist:
                    tmpInd = j
                    minDist = newDist
            v2v_receivers[i] = tmpInd
        # 从剩下的车里面随机选择一些作为CUE
        cntCUE = numV2V + 1
        # 剩下的车辆选择numV2V个为V2V的发送端
        indCUE = []
        while cntCUE <= len(self.vehicles):
            if indvehicles[cntCUE] not in v2v_receivers:
                indCUE.append(indvehicles[cntCUE])
            cntCUE += 1
            if len(indCUE) >= numV2I:
                break
        return indCUE, v2v_senders, v2v_receivers


# def classify_direction(self):
    #     # Initialize lists for each direction
    #     up_vehicles = []
    #     down_vehicles = []
    #     left_vehicles = []
    #     right_vehicles = []
    #     # Loop through each vehicle in the environment and append to the correct list based on direction
    #     for i in len(self.vehicles):
    #         if self.vehicles[i].direction == 'up':
    #             up_vehicles.append(self.vehicles.)
    #         elif vehicle.direction == 'down':
    #             down_vehicles.append(vehicle)
    #         elif vehicle.direction == 'left':
    #             left_vehicles.append(vehicle)
    #         elif vehicle.direction == 'right':
    #             right_vehicles.append(vehicle)
    #
    #     # Print the number of vehicles in each list for verification
    #     print("Up vehicles:", len(up_vehicles))
    #     print("Down vehicles:", len(down_vehicles))
    #     print("Left vehicles:", len(left_vehicles))
    #     print("Right vehicles:", len(right_vehicles))


if __name__ == "__main__":
    up_lanes = [i / 2.0 for i in
                [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
    down_lanes = [i / 2.0 for i in
                  [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
                   750 - 3.5 / 2]]
    left_lanes = [i / 2.0 for i in
                  [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
    right_lanes = [i / 2.0 for i in
                   [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                    1299 - 3.5 / 2]]

    width = 750 / 2
    height = 1298 / 2
    numV2I = 10
    numV2V = 10

    env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height)
    num_vehicles = 100
    velocitys, start_poss = env.generate_vehicles(num_vehicles)
    indCUE, v2v_senders, v2v_receivers = env.generate_V2I_and_V2V(numV2I, numV2V)
    print(indCUE, v2v_senders, v2v_receivers)


    # print(velocitys)
    # print(start_poss)
    # do something with the generated vehicles
