import numpy as np
from Function import Fun
import math
import warnings
warnings.filterwarnings("ignore")


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    Input = xtrain
    Target = ytrain
    UR = 0.3
    UR_Max = 0.3
    UR_Min = 0.001
    Max_Run = 1
    Run = 1
    Max_FEs = opts['T'] * opts['N']
    Cost = np.zeros([Max_Run, Max_FEs])
    FN = np.zeros([Max_Run, Max_FEs])
    arr = np.random.randint(0, 2, 10)

    while (Run <= Max_Run):
        EFs = 1

        X = np.random.randint(0, 2, np.size(Input, 1))  # Initialize an Individual X
        Fit_X = Fun(xtrain, xvalid, ytrain, yvalid, X, opts)  # Calculate the Fitness of X
        Nvar = np.size(Input, 1)  # Number of Features in Dataset

        while (EFs <= Max_FEs):

            X_New = np.copy(X)
            # Non-selection operation:

            U_Index = np.where(X == 1)  # Find Selected Features in X
            NUSF_X = np.size(U_Index, 1)  # Number of Selected Features in X
            UN = math.ceil(UR * Nvar)  # The Number of Features to Unselect: Eq(2)
            # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
            # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
            K1 = np.random.randint(0, NUSF_X,
                                   UN)  # Generate UN random number between 1 to the number of slected features in X
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]  # K=index(U)
            X_New[K] = 0  # Set X_New (K)=0

            # Selection operation:
            if np.sum(X_New) == 0:
                S_Index = np.where(X_New == 0)  # Find non-selected Features in X
                NSF_X = np.size(S_Index, 1)  # Number of non-selected Features in X
                SN = 1  # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X,
                                       SN)  # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                X_New = np.copy(X)
                X_New[K] = 1  # Set X_New (K)=1

            Fit_X_New = Fun(xtrain, xvalid, ytrain, yvalid, X_New, opts)  # Calculate the Fitness of X_New

            if Fit_X_New < Fit_X:
                X = np.copy(X_New)
                Fit_X = Fit_X_New

            UR = (UR_Max - UR_Min) * ((Max_FEs - EFs) / Max_FEs) + UR_Min  # Eq(3)
            Cost[ Run - 1, EFs - 1] = Fit_X
            # FN[EFs - 1, Run - 1] = np.sum(X)
            # print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format(EFs, Fit_X,
            #                                                                                                np.sum(X),
            #                                                                                                Run))
            EFs = EFs + 1
        Run = Run + 1
    # 计算需要等差选取的数据点个数
    num_points = opts['T']
    # 等差选取数据点的索引
    indices = np.round(np.linspace(0, Max_FEs - 1, num_points)).astype(int)
    # 从Cost中选取数据填入curve
    curve = Cost[:, indices]
    # 将curve变形为所需的大小
    curve = curve.reshape(1, -1)
    # 现在curve中包含了从Cost中等差选取的数据
    sfe_data = {'sf': X, 'c': curve, 'nf': np.sum(X)}
    return sfe_data