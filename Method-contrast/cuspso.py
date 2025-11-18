import copy
import math

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import KNeighborsClassifier
from staticFunction import *
from newMethod.superDemo import superDemo
from skrebate import ReliefF

import numpy as np
from numpy.random import rand
from Function import Fun


class Particle(superDemo):
    def __init__(self, feature, target, PopSize=30, Threshold=0.6, Nc=2, w=0.1, c1=1.49445, c2=1.49445, r1=0.1, r2=0.1):
        super(Particle, self).__init__(feature, target)
        self.gBestAcc = 1
        self.PopSize = PopSize
        self.Threshold = Threshold
        self.Nc = Nc
        self.dim = feature.shape[1]
        self.positionPre = np.zeros((PopSize, self.dim))
        self.positionNew = np.zeros((PopSize, self.dim))
        self.velocityNew = np.zeros((PopSize, self.dim))
        self.velocityPre = np.zeros((PopSize, self.dim))
        self.pArrayNew = np.zeros((PopSize, self.dim))
        self.pArrayPre = np.zeros((PopSize, self.dim))
        self.accuracyNew = np.zeros(PopSize)
        self.accuracyPre = np.zeros(PopSize)
        self.selectDimNew = np.zeros(PopSize)
        self.selectDimPre = np.zeros(PopSize)
        self.fitnessPre = np.zeros(PopSize)
        self.fitnessNew = np.zeros(PopSize)
        self.CPosition = np.zeros((PopSize * (Nc + 1), self.dim))
        self.CVelocity = np.zeros((PopSize * (Nc + 1), self.dim))
        self.CFitness = np.zeros(PopSize * (Nc + 1))
        self.SPosition = np.zeros((PopSize, self.dim))
        self.SVelocity = np.zeros((PopSize, self.dim))
        self.SFitness = np.zeros(PopSize)
        self.pBest = np.zeros((PopSize, self.dim))
        self.pBestFit = np.ones(PopSize)
        self.gBest = np.zeros(self.dim)
        self.gBestArr = np.zeros(self.dim)
        self.gBestFit = 1
        self.probability = np.zeros(self.dim)
        self.A = 0.15
        self.B = 0.05
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2

    def run(self, iteration):
        self.initial()
        self.weight_reliefF()
        for i in range(iteration):
            self.w = 0.9 - 0.5 * (i / iteration)
            self.renew_position()
            self.correlation_guid_updating()
            self.correlation_fitness()
            self.particle_selection_strategy()
            print(f"第{i}代适应度{self.gBestFit}")
            print(f"第{i}代准确度{self.gBestAcc}")

    def initial(self):
        self.positionNew = np.random.rand(self.PopSize, self.dim)
        self.pArrayNew = self.pos_to_onehot(self.positionNew)
        self.compute_fitness(self.pArrayNew)
        self.renew_position()

    # 按标准方式更新位置
    def renew_position(self):
        fitness_sort_idx = np.argmin(self.fitnessNew)
        for i in range(self.PopSize):
            if self.fitnessNew[i] < self.pBestFit[i]:
                self.pBest[i, :] = copy.copy(self.positionNew[i, :])
                self.pBestFit[i] = self.fitnessNew[i]
        if self.gBestFit > self.fitnessNew[fitness_sort_idx]:
            self.gBest = copy.copy(self.positionNew[fitness_sort_idx, :])
            self.gBestFit = self.fitnessNew[fitness_sort_idx]
            self.gBestArr = copy.copy(self.pArrayNew[fitness_sort_idx, :])
            self.gBestAcc = self.accuracyNew[fitness_sort_idx]
        self.SPosition = copy.copy(self.positionNew)
        self.SVelocity = copy.copy(self.velocityNew)
        self.SFitness = copy.copy(self.fitnessNew)
        self.positionPre = copy.copy(self.positionNew)
        self.r1 = np.random.rand()
        self.r2 = np.random.rand()
        self.velocityNew = self.w * self.velocityPre + self.c1 * self.r1 * (self.pBest - self.positionPre) \
                           + self.c2 * self.r2 * (self.gBest - self.positionPre)
        self.positionNew = self.positionPre + self.velocityNew
        self.positionNew = np.clip(self.positionNew, 0, 1)

    # 利用reliefF方式计算特征与标签的得分以及每个特征的转换概率
    def weight_reliefF(self):
        fs = ReliefF(n_neighbors=100)
        X_selected = fs.fit_transform(self.feature, self.target)
        feature_scores = np.array(fs.feature_importances_)
        score_min = feature_scores.min()
        score_max = feature_scores.max()
        weight = (feature_scores - score_min) / (score_max - score_min)
        self.probability = self.A * np.sin(weight * math.pi) + self.B

    # 根据correlation策略产生新的粒子
    def correlation_guid_updating(self):
        self.CPosition[:self.PopSize, :] = copy.copy(self.SPosition)
        self.CVelocity[:self.PopSize, :] = copy.copy(self.SVelocity)
        for i in range(self.Nc):
            self.change_position(i, self.positionNew)

    # 转换粒子
    def change_position(self, Ni, position):
        for pop in range(self.PopSize):
            for i in range(self.dim):
                if np.random.rand() < self.probability[i]:
                    self.CPosition[pop + (Ni + 1) * self.PopSize, i] = 1 - position[pop, i]
                self.CVelocity[pop + (Ni + 1) * self.PopSize, :] = copy.copy(self.SVelocity[pop, :])

    def correlation_fitness(self):
        distance_S = np.zeros((self.PopSize * (self.Nc + 1), self.PopSize))
        for i in range(self.PopSize * (self.Nc + 1)):
            for j in range(self.PopSize):
                distance_S[i, j] = self.distance(self.CPosition[i, :], self.SPosition[j, :])
        sortDistance_S = np.argsort(distance_S)
        for i in range(self.PopSize * (self.Nc + 1)):
            self.CFitness[i] = np.mean(self.SFitness[sortDistance_S[:3]])

    def particle_selection_strategy(self):
        CFitness = copy.copy(self.CFitness)
        CPosition = copy.copy(self.CPosition)
        CVelocity = copy.copy(self.CVelocity)
        position = np.zeros((self.PopSize, self.dim))
        velocity = np.zeros((self.PopSize, self.dim))
        for i in range(self.PopSize):
            min_CFitness_idx = CFitness.argmin()
            CFitness_min = CFitness[min_CFitness_idx]
            CPosition_min = CPosition[min_CFitness_idx, :]
            CVelocity_min = CVelocity[min_CFitness_idx, :]
            CFitness = np.delete(CFitness, min_CFitness_idx, axis=0)
            CPosition = np.delete(CPosition, min_CFitness_idx, axis=0)
            CVelocity = np.delete(CVelocity, min_CFitness_idx, axis=0)
            distance_arr = np.zeros(CFitness.shape[0])
            for j in range(CFitness.shape[0]):
                distance_arr[j] = self.distance(CPosition[j, :], CPosition_min)
            nearst_idx = distance_arr.argmin()
            CFitness = np.delete(CFitness, nearst_idx, axis=0)
            CPosition = np.delete(CPosition, nearst_idx, axis=0)
            CVelocity = np.delete(CVelocity, nearst_idx, axis=0)
            position[i, :] = CPosition_min
            velocity[i, :] = CVelocity_min
        self.positionNew = copy.copy(position)
        self.velocityNew = copy.copy(position)
        self.pArrayPre = copy.copy(self.pArrayNew)
        self.pArrayNew = self.pos_to_onehot(self.positionNew)
        self.fitnessPre = copy.copy(self.fitnessNew)
        self.compute_fitness(self.pArrayNew)

    def compute_fitness(self, pArray):
        for i in range(self.PopSize):
            index = np.where(pArray[i, :])[0]
            if index.shape[0] <= 0:
                self.accuracyNew[i] = 0
                self.selectDimNew[i] = 0
                self.fitnessNew[i] = self.fitness(self.accuracyNew[i], self.selectDimNew[i], self.dim)
            else:
                self.classify(index)
                self.accuracyNew[i] = self.accuracy()
                self.selectDimNew[i] = np.sum(pArray[i, :])
                self.fitnessNew[i] = self.fitness(self.accuracyNew[i], self.selectDimNew[i], self.dim)


    def pos_to_onehot(self, position):
        arr = np.zeros((self.PopSize, self.dim))
        arr[position > self.Threshold] = 1
        return arr

    @staticmethod
    def fitness(acc, featureLen, dim):
        return 0.9 * (1 - acc) + 0.1 * featureLen / dim

    # 计算两个粒子的欧式距离
    @staticmethod
    def distance(p1, p2):
        return np.sum(np.abs(p1 - p2))


import numpy as np
from numpy.random import rand
from Function import Fun

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
    return X

def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]
    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()
    return V, Vmax, Vmin

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # Parameters
    ub = 1
    lb = 0
    thres = 0.5
    w = 0.9  # inertia weight
    c1 = 2  # acceleration factor
    c2 = 2  # acceleration factor
    N = opts['N']
    max_iter = opts['T']
    if 'w' in opts:
        w = opts['w']
    if 'c1' in opts:
        c1 = opts['c1']
    if 'c2' in opts:
        c2 = opts['c2']
        # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    gBestAcc = 1
    PopSize = N
    Threshold = Threshold
    Nc = Nc
    dim = feature.shape[1]
    positionPre = np.zeros((PopSize, dim))
    positionNew = np.zeros((PopSize, dim))
    velocityNew = np.zeros((PopSize, dim))
    velocityPre = np.zeros((PopSize, dim))
    pArrayNew = np.zeros((PopSize, dim))
    pArrayPre = np.zeros((PopSize, dim))
    accuracyNew = np.zeros(PopSize)
    accuracyPre = np.zeros(PopSize)
    selectDimNew = np.zeros(PopSize)
    selectDimPre = np.zeros(PopSize)
    fitnessPre = np.zeros(PopSize)
    fitnessNew = np.zeros(PopSize)
    CPosition = np.zeros((PopSize * (Nc + 1), dim))
    CVelocity = np.zeros((PopSize * (Nc + 1), dim))
    CFitness = np.zeros(PopSize * (Nc + 1))
    SPosition = np.zeros((PopSize, dim))
    SVelocity = np.zeros((PopSize, dim))
    SFitness = np.zeros(PopSize)
    pBest = np.zeros((PopSize, dim))
    pBestFit = np.ones(PopSize)
    gBest = np.zeros(dim)
    gBestArr = np.zeros(dim)
    gBestFit = 1
    probability = np.zeros(dim)
    A = 0.15
    B = 0.05
    # w = w
    # c1 = c1
    # c2 = c2
    # r1 = r1
    # r2 = r2

