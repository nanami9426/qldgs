import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from Function import Fun

import warnings
warnings.filterwarnings('ignore')


class HPSO_LS():
    def __init__(self, x, y, xtrain, xvalid, ytrain, yvalid, opts, NC=50, NP=20, c1=2, c2=2, v_max=4, v_min=-4, a=0.65, seed=89):

        # # set x_train,x_test,y_train and y_test
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
        # # normalize data
        scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        self.X_train = scaler.fit_transform(xtrain)
        self.X_test = scaler.fit_transform(xvalid)
        self.y_train = ytrain
        self.y_test = yvalid
        self.opts = opts
        self.x = x
        self.curve = []

        self.feature = x.columns  # features of dataset
        self.NC = opts['T']  # number of iterations
        self.NP = opts['N']   # number of particles
        self.c1 = c1  # learning rate1
        self.c2 = c2  # learning rate2
        self.v_max = v_max  # v max
        self.v_min = v_min  # v min
        self.a = a  # alpha
        self.k = None  # number of determine in first step
        self.similar = []  # similar features
        self.dissimilar = []  # dissimilar features
        self.seed = seed  # random seed

    def determining_k(self):  # step1: determining k(numbevr of features)
        random.seed(self.seed)  # set random seed
        f = len(self.feature)  # number of features
        l_sf = []  # probability of each sf
        # implement l_sf formula
        for sf in range(1, f):
            l = f - sf
            s = 0  # sum
            for i in range(l):  # sigma
                i += 1
                s += f - i
            l_sf.append(round(l / s, 2))
        M = 0  # max of random numbers
        while (M <= 2):
            e = (random.randint(15, 70)) / 100  # epsilon
            M = int(e * f)  # max of random numbers
            # k = random.choices(list(range(3, M + 1)), weights=l_sf[3:M + 1])  # roulette wheel
            ### 修改
            if max(l_sf[3:M + 1]) <= 0:
                k = 3
            else:
                k = random.choices(list(range(3, M + 1)), weights=l_sf[3:M + 1])  # roulette wheel
        self.k = k[0]  # set k
        return k[0]

    def grouping_features(self, plot=False):  # step 2

        if plot:  # plot the correlation if true
            corr = self.x.corr()
            plt.figure(figsize=(10, 10))
            sns.heatmap(corr, annot=True)

        corr = []  # to keep corrlations
        f = len(self.feature)  # number of features
        for i in range(1, f):
            corr_i = self.x.corrwith(self.x[i])  # pearson corr for each i
            corr_i[i] = 0  # In order not to be effective in sigma
            corr.append(((round(corr_i.apply(abs).sum() / (f - 1), 3)), i))  # caculate corr for each i
            corr.sort()  # sort
        # seperate to similar and dissimilar
        self.dissimilar = corr[:len(corr) // 2]
        self.similar = corr[len(corr) // 2:]
        # print("similar:   ", self.similar, "\ndissimilar :  ", self.dissimilar)

    def initializing_particles(self):  # step 3

        random.seed(self.seed)  # set random seed
        f = len(self.feature)  # number of features
        k = self.determining_k()  # determining number of features for selection
        particles = []  # list of particles
        sample = [1] * k + [0] * (f - k)  # a sample particle to cahnge later
        for i in range(self.NP):
            random.shuffle(sample)  # shuffling the sample
            change = sample.copy()  # copy
            particles.append(change)  # add to particles list
        velocity = [[round(random.random(), 3) for x in range(f)] for y in
                    range(self.NP)]  # set a random num in [0,1] for velocity of each dim of particles

        return particles, velocity

    def fitness(self, P):  # step 6
        score = []  # list of scores
        for p in P:
            # model = KNeighborsClassifier()  # make a knn model
            # params = {
            #     'n_neighbors': [1]  # k in knn
            # }
            # grid_model = GridSearchCV(model, params, cv=10)  # cross validation by grid search
            #
            # # set the x according to the selected features
            # X_tr = pd.DataFrame(self.X_train)[[j for i, j in zip(p, pd.DataFrame(self.X_train).columns) if i == 1]]
            # X_tst = pd.DataFrame(self.X_test)[[j for i, j in zip(p, pd.DataFrame(self.X_test).columns) if i == 1]]
            # grid_model.fit(X_tr, self.y_train)  # train
            # y_pred = grid_model.predict(X_tst)  # test
            # score.append(round(accuracy_score(self.y_test, y_pred), 3))  # fitness
            Xbin = np.array(p)
            xtrain = self.X_train
            xvalid = self.X_test
            ytrain = self.y_train
            yvalid = self.y_test
            fit = Fun(xtrain, xvalid, ytrain, yvalid, Xbin, self.opts)
            score.append(round(1-fit, 3))
            self.curve.append(fit)

        return score

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update(self, P, V, p_best, g_best):  # step 4
        f = len(self.feature)  # number of features
        v_new = np.zeros_like(V)  # in order to keep the new vlocities
        p_new = np.zeros_like(P)  # in order to keep the new particles

        for i in range(self.NP):  # for each particle
            for d in range(f):  # for each dimention
                v_new[i][d] = V[i][d] + self.c1 * random.random() * (p_best[i][d] - P[i][d]) * V[i][
                    d] + self.c2 * random.random() * (g_best[d] - P[i][d])  # v(t+1) formula
                # if v is less than v_min it's v_min
                if v_new[i][d] < self.v_min:
                    v_new[i][d] = self.v_min
                # if v is more than v_max it's v_max
                if v_new[i][d] > self.v_max:
                    v_new[i][d] = self.v_max

                rand = random.random()  # set the rand in formula of x(t+1)
                # formual of x(t+1)
                if self.sigmoid(v_new[i][d]) > rand:
                    p_new[i][d] = 1
                else:
                    p_new[i][d] = 0

        return p_new, v_new

    def local_search(self, P):  # step 5
        ns = int(self.a * self.k)  # calculate ns
        nd = int((1 - self.a) * self.k)  # calculate nd
        p_new = []  # for new particles
        for particle in range(self.NP):
            # lists for seperate similar and dissimilars
            x_d = []  # X_d
            x_s = []  # X_s
            features = [j for j, i in enumerate(P[particle]) if i == 1]  # a set of selected features
            # seperate the features into similar and dissimilar
            for i, j in self.similar:
                if j in features:
                    x_s.append((i, j))
            for i, j in self.dissimilar:
                if j in features:
                    x_d.append((i, j))

            l = ns - len(x_s)  # The difference between ns and xs
            if l > 0:  # we should add
                for i in range(l):
                    for s in self.similar:
                        if s not in x_s:  # similars are sorted so we choose the first one that we don't selected
                            x_s.append(s)
                            break
            elif l < 0:  # we should delete
                for i in range(-l):
                    for x in x_s[::-1]:
                        x_s.pop()  # the list is sorted so we pop the last one
                        break

            l = nd - len(x_d)  # The difference between nd and xd
            if l > 0:  # we should add
                for i in range(l):
                    for d in self.dissimilar:
                        if d not in x_d:
                            x_d.append(d)  # dissimilars are sorted so we choose the first one that we don't selected
                            break
            elif l < 0:  # we should delete
                for i in range(-l):
                    for x in x_d[::-1]:
                        x_d.pop()  # the list is sorted so we pop the last one
                        break
                        # we keep the i and corr_i, we don't need the corr_i any more so delete it
            x_s = [i[1] for i in x_s]
            x_d = [i[1] for i in x_d]
            x = x_d + x_s  # all the selected features
            sample = [0] * len(self.feature)  # sample
            # set the new particles according to selected featurs
            for i in x:
                sample[i] = 1
            particle = sample.copy()
            p_new.append(particle)

        return p_new

    def Run_HPSO_LS(self):
        self.grouping_features()  # step 2
        P, V = self.initializing_particles()  # step 1 and step 3
        scores = self.fitness(P)  # calculate fitness for particles
        p_best = [(P[i], scores[i]) for i in range(len(P))]  # p_best for iter=1
        g_best = (P[scores.index(max(scores))], max(scores))  # g_best and its score
        # print(g_best, "g_best")
        for i in range(self.NC):
            p_new, v_new = self.update(P, V, [i[0] for i in p_best], g_best[0])  # x(t+1) and v(t+1)
            p_new = self.local_search(p_new)  # step 5
            scores_new = self.fitness(p_new)  # step 6

            best_p = []
            # update g_best and p_best
            for p in range(self.NP):
                if scores[p] > scores_new[p]:
                    best_p.append((P[p], scores[p]))
                else:
                    best_p.append((p_new[p], scores_new[p]))

                if scores_new[p] > g_best[1]:
                    g_best = [p_new[p], scores_new[p]]
                    # print(g_best, "new g_best")

            P = p_new  # update particles
            V = v_new  # update particles
            scores = scores_new.copy()  # update p_best scores
            p_best = best_p.copy()  # update p_best

        return g_best


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    x = np.concatenate((xtrain, xvalid), axis=0)
    y = np.concatenate((ytrain, yvalid), axis=0)
    x  = pd.DataFrame(x )
    A = HPSO_LS(x=x, y=y, xtrain=xtrain, xvalid=xvalid, ytrain=ytrain, yvalid=yvalid, opts=opts)
    features, accuracy = A.Run_HPSO_LS()
    Gbin = np.array(features)
    c = np.array(A.curve)
    sorted_idx = np.argsort(c)[::-1]  # 对 c 中的元素进行排序，并返回排序后的索引
    sorted_c = c[sorted_idx]  # 根据排序后的索引，获取对应的数值
    num_points = opts['T']
    indices = np.round(np.linspace(0, np.size(sorted_c, 0) - 1, num_points)).astype(int)
    curve = sorted_c[indices]
    curve = curve.reshape(1, -1)
    HPSOLS_data = {'sf': Gbin, 'c': curve, 'nf': np.sum(Gbin)}
    return HPSOLS_data