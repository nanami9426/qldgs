import numpy as np
from Function import Fun
# import PyIFS
import math

'''
输入：
x是 T 乘以 n 矩阵，其中 T 是样本数，n 是特征数
y是带有类标签的列向量
alpha是混合参数
supervision是一个布尔变量（0 = 无监督版本，1 = 监督版本）
verbose是一个布尔变量 （0， 1）可以控制在执行这些步骤时是否打印出相应的消息，使得代码执行的过程更易于理解。
输出：
RANKED是按属性重要性排序的列索引x
WEIGHT是属性权重，具有分配给重要属性的大正权重
'''
import numpy as np
from scipy import stats
import math

import numpy as np
from scipy import stats


class InfFS:
    def __init__(self, full_data):
        """初始化时预计算完整特征集的相关系数矩阵"""
        self.full_data = full_data
        self.full_corr = self._compute_full_correlation(full_data)

    def _compute_full_correlation(self, data):
        """计算并预处理完整特征集的相关系数矩阵"""
        corr_matrix, _ = stats.spearmanr(data)
        return np.nan_to_num(corr_matrix, nan=0.0)  # 处理缺失值

    def _get_submatrix(self, feature_mask):
        """从完整矩阵中提取特征子集的相关系数矩阵"""
        indices = np.where(feature_mask == 1)[0]
        lists = [[self.full_corr[i, j] for j in indices] for i in indices]
        return np.array(lists)

    def infFS(self, X_bin, alpha, verbose=False):
        """优化后的特征评分方法"""
        # 获取特征子矩阵
        corr_ij = self._get_submatrix(X_bin)

        # 计算标准差矩阵（基于当前特征子集）
        selected_index = np.where(X_bin == 1)[0]
        selected_data = self.full_data[:, selected_index]
        STD = np.std(selected_data, ddof=1, axis=0)

        # 构建标准差矩阵
        STDMatrix = self.bsxfun(STD)
        STDMatrix = self.SubtractMin(STDMatrix)
        sigma_ij = self.DivideByMax(STDMatrix)
        for i in range(0, sigma_ij.shape[0]):
            for j in range(0, sigma_ij.shape[1]):
                if math.isnan(sigma_ij[i, j]) or sigma_ij[i, j] < -1 or sigma_ij[i, j] > 1:
                    sigma_ij[i, j] = 0

        # 后续计算流程
        A = alpha * corr_ij + (1 - alpha) * sigma_ij
        I = np.identity(A.shape[0])
        r = 0.9 / np.max(np.abs(np.linalg.eigvals(A)))
        S = np.linalg.inv(I - r * A) - I

        WEIGHT = np.sum(S, axis=1)
        RANKED = np.argsort(WEIGHT)
        RANKED = np.flip(RANKED, 0)
        RANKED = RANKED.T
        WEIGHT = WEIGHT.T
        return RANKED, WEIGHT

    # 以下为优化后的工具方法
    def bsxfun(self, STD):
        return np.maximum.outer(STD, STD)

    def SubtractMin(self, matrix):
        return matrix - np.min(matrix)

    def DivideByMax(self, matrix):
        return matrix / np.max(matrix) if np.max(matrix) != 0 else matrix

    def Fun(self, x_train, X_bin, alpha, verbose):
        RANKED, WEIGHT = self.infFS(X_bin, alpha)
        dim = len(np.where(X_bin == 1)[0])
        # dim = x_train.shape[1]
        fit = -np.sum(WEIGHT) / dim
        return fit
