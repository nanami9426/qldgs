import numpy as np
import pandas as pd
from Function import Fun
# import PyIFS
import math
from scipy import stats
# from mifs import MRMR

from sklearn.feature_selection import mutual_info_classif
# from mrmr import mrmr_classif
import pymrmr
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


def calculate_mrmr_scores(X, y, method='MIQ'):
    n_features = X.shape[1]
    relevance = mutual_info_classif(X, y)  # 特征与标签的相关性
    redundancy = np.zeros(n_features)

    # 计算每个特征与其他所有特征的平均冗余
    for i in range(n_features):
        others = [j for j in range(n_features) if j != i]
        redundancy[i] = np.mean([mutual_info_classif(X[:, i].reshape(-1, 1), X[:, j])[0] for j in others])

    if method == 'MIQ':
        scores = relevance / (redundancy + 1e-6)  # 避免除零
    else:
        scores = relevance - redundancy

    return scores
def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # inf
    x = np.concatenate((xtrain, xvalid), axis=0)
    y = np.concatenate((ytrain, yvalid), axis=0)
    supervision = 1
    verbose = 0
    alpha = 0.5
    # inf = PyIFS.InfFS()
    # [RANKED, WEIGHT] = inf.infFS(x, y, alpha, supervision, verbose)
    # 使用 mRMR 方法进行特征选择
    # 计算特征与目标变量的互信息
    # mi = mutual_info_classif(x, y)
    #
    # # 计算特征之间的互信息矩阵
    # n_features = x.shape[1]
    # redundancy = np.zeros(n_features)
    # for i in range(n_features):
    #     for j in range(n_features):
    #         if i != j:
    #             redundancy[i] += mutual_info_classif(x[:, i], x[:, j])[0]
    # X为特征矩阵（n_samples×n_features），y为标签
    # score = MRMR.mrmr(X, y, n_selected_features=10, method="MIQ")
    # selected_indices = np.argsort(score)[::-1][:10]
    # # 计算 mRMR 评分
    # mrmr_scores = mi - alpha * redundancy
    #
    # # 对特征按评分进行排序
    # RANKED = np.argsort(-mrmr_scores)
    #
    # # model = MRMR(method='MIQ', k=10)
    # # model.fit(X, y)

    # mrmr_scores = calculate_mrmr_scores(x, y, method='MIQ')
    # RANKED = np.argsort(-mrmr_scores)

    x_pd =  pd.DataFrame(x)
    y_pd = pd.DataFrame(y)
    df = pd.concat([x_pd, y_pd])
    selected_features = pymrmr.mRMR(df, method='MIQ', K=x.sahpe[1])

    dim = np.size(xtrain, 1)
    N = opts['N']
    max_iter = opts['T']
    curve = np.zeros([1, max_iter], dtype='float')
    Xpb = np.zeros([max_iter, dim], dtype='int')
    fitP = float('inf') * np.ones([max_iter, 1], dtype='float')
    Xgb = np.zeros([1, dim], dtype='int')
    fitG = float('inf')
    for i in range(max_iter):
        sf = RANKED[:max_iter-i]
        Xpb[i,sf] = 1
        fitP[i, 0] = Fun(xtrain, xvalid, ytrain, yvalid, Xpb[i, :], opts)
        if fitP[i, 0] <= fitG:
            Xgb[0, :] = Xpb[i, :]
            fitG = fitP[i, 0]
        curve[0, i] = fitG.copy()
    num_feat = np.sum(Xgb[0, :])

    inf_data = {'sf': Xgb, 'c': curve, 'nf': num_feat}

    return inf_data



class InfFS:

    def __init__(self):
        h=0.1
        #print("HI")
    # Take in input the matrix e the label vector and return a matrix
    # of data for every different label.
    def takeLabel(self, x_train, y_train ):
        counter = x_train.shape[0] -1
        s_n = x_train
        s_p = x_train
        while(1):
            if( y_train[counter] == 1 ):
                s_n = np.delete(s_n, counter, axis = 0 )
            else:
                s_p = np.delete(s_p, counter, axis = 0 )
            counter = counter - 1
            if( counter == - 1 ):
                break
        return s_p, s_n

    # Function that help to define priors_corr.
    def defPriorsCorr(self,mu_s_n, mu_s_p):
        pcorr = mu_s_p
        counter = 0
        while( counter < len(pcorr) ):
            pcorr[counter] = (pcorr[counter] - mu_s_n[counter])*(pcorr[counter] - mu_s_n[counter])
            counter = counter + 1
        return pcorr

    # Function to subtract the min value of the matrix to all it's elements.
    def SubtractMin(self, corr_ij ):
        m = 10100
        for i in range(0,corr_ij.shape[0]): # Find the min.
            for j in range(0,corr_ij.shape[1]):
                if( corr_ij[i,j] < m ):
                    m = corr_ij[i,j]

        for i in range(0,corr_ij.shape[0]): # Subtract the min value.
            for j in range(0,corr_ij.shape[1]):
                corr_ij[i,j] = corr_ij[i,j] - m

        return corr_ij

    # Function to divide every element of the matrix to his maximum value.
    def DivideByMax(self,corr_ij):
        m = -1
        for i in range(0,corr_ij.shape[0]): # Find the max.s
            for j in range(0,corr_ij.shape[1]):
                if( corr_ij[i,j] > m ):
                    m = corr_ij[i,j]

        for i in range(0,corr_ij.shape[0]): # Divide by the maximum value.
            for j in range(0,corr_ij.shape[1]):
                corr_ij[i,j] = corr_ij[i,j] / m

        return corr_ij

    # Handmaded bsxfunction that take the max.
    def bsxfun(self, STD ):
        m = np.zeros( (STD.shape[0], STD.shape[0]) )
        for i in range( 0,STD.shape[0] ):
            for j in range( 0,STD.shape[0] ):
                if( STD[i] > STD[j] ):
                    m[i,j] = STD[i]
                else:
                    m[i,j] = STD[j]
        return m

    def infFS(self,x_train, y_train, alpha, supervision, verbose):
        # Start of point one.
        # supervision = 0
        if supervision:
            s_p, s_n = self.takeLabel( x_train, y_train)
            mu_s_n = s_n.mean(0)
            mu_s_p = s_p.mean(0)
            priors_corr = self.defPriorsCorr(mu_s_n, mu_s_p)
            st = np.power(np.std(s_p, ddof = 1, axis = 0),2)
            st = st + np.power(np.std(s_n,ddof = 1, axis = 0),2)
            for i in range(0,len(st)):
                if st[i] == 0:
                    st[i] = 10000
            corr_ij = priors_corr
            for i in range(0,len(corr_ij)):
                corr_ij[i] = corr_ij[i] / st[i]
            corr_ij = np.dot( corr_ij.T[:,None], corr_ij[None,:])
            corr_ij = self.SubtractMin(corr_ij)
            corr_ij = self.DivideByMax(corr_ij)
        else:
            corr_ij, pval = stats.spearmanr(x_train)
            for i in range( 0,corr_ij.shape[0] ):
                for j in range( 0,corr_ij.shape[1] ):
                    if( math.isnan(corr_ij[i,j]) or corr_ij[i,j] < -1 or corr_ij[i,j] > 1 ):
                        corr_ij[i,j] = 0

        # After if.
        STD = np.std(x_train, ddof = 1, axis = 0)
        STDMatrix = self.bsxfun( STD )
        STDMatrix = self.SubtractMin(STDMatrix)
        sigma_ij = self.DivideByMax(STDMatrix)
        for i in range( 0,sigma_ij.shape[0] ):
            for j in range( 0,sigma_ij.shape[1] ):
                if( math.isnan(sigma_ij[i,j]) or sigma_ij[i,j] < -1 or sigma_ij[i,j] > 1 ):
                    sigma_ij[i,j] = 0

        # End of point one.

        # Start of the point two.
        if (verbose):
            print("2) Building the graph G = <V,E> \n");
        A =  ( alpha*corr_ij + (1-alpha)*sigma_ij );
        # End of the point two.

        # Start of the point three.
        if (verbose):
            print("3) Letting paths tend to infinite \n");

        I = np.identity( A.shape[0] )
        r = ( 0.9/ max( np.linalg.eigvals(A) ) ) # Setting the r values.
        y = I - ( r * A )
        S = np.linalg.inv( y ) - I
        # End of point three.

        # Start of point four.
        if (verbose):
            print("4) Estimating energy scores \n")

        WEIGHT = np.sum( S , axis=1 )
        # End of point four.

        # Start of point five.
        if(verbose):
            print("5) Features ranking")

        RANKED = np.argsort(WEIGHT)
        RANKED = np.flip(RANKED,0)
        RANKED = RANKED.T
        WEIGHT = WEIGHT.T
        return RANKED, WEIGHT
        # End of point five.