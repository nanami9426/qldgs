from SAGA2 import EvolutionaryWrapperFeatureSelection
import time
import pandas as pd
import numpy as np
import copy
import random
from deap import tools
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from Function import Fun
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score


class ProcessingDataset:

    # The init method or constructor
    def __init__(self, xtrain, xvalid, ytrain, yvalid, opts, label, divide_dataset=True, header=None):
        X = np.concatenate((xtrain, xvalid), axis=0)
        y = np.concatenate((ytrain, yvalid), axis=0)
        df = pd.DataFrame(X)
        df['labels'] = pd.Series(y)
        # Set raw data attribute
        self.df = df
        self.df_sampled = df
        self.label = label
        self.data = {'xtrain': xtrain, 'xvalid': xvalid, 'ytrain': ytrain, 'yvalid':yvalid}
        self.opts = opts
        self.curve = []

        # Encode categorical features
        le = preprocessing.LabelEncoder()
        for col in self.df.columns:
            if (self.df[col].dtypes == 'object'):
                list_of_values = list(self.df[col].unique())
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
                le.fit(list_of_values)
                self.df[col] = le.transform(self.df[col])

        # Replcae missing values
        self.df = self.df.replace('?', np.NaN)
        self.df = self.df.fillna(self.df.median())
        # df = df.astype('int32')

        if (divide_dataset):
            self.divideDataset()

    def divideDataset(self, classifier, normalize=True, shuffle=True, all_features=True, all_instances=True,
                      evaluate=True, partial_sample=False, folds=10):

        # Set classifier
        self.clf = copy.copy(classifier)
        self.folds = folds

        # Shuffle dataset
        if (shuffle):
            self.df = self.df.sample(frac=1)
            self.df_sampled = self.df

        # Sample from dataset
        if (partial_sample):
            self.df_sampled = self.df.sample(n=partial_sample)

        # Divide datset into training/validation/testing
        if (self.label == -1):
            self.X = self.df_sampled.iloc[:, :-1].values
            self.y = self.df_sampled.iloc[:, -1].values
        # else:
        #     selector = [x for x in range(self.df.shape[1]) if x != label]
        #     self.X = self.df_sampled.iloc[:, selector].values
        #     self.y = self.df_sampled.iloc[:, label].values

        # X_train = self.X[0:int(0.6 * len(self.X)), :]
        # mean = X_train.mean(axis=0)
        # std = X_train.std(axis=0) + 0.0001
        # X_train_normalized = (X_train - mean) / std
        # y_train = self.y[0:int(0.6 * len(self.X))]
        #
        # X_val = self.X[int(0.6 * len(self.X)):int(0.8 * len(self.X)), :]
        # y_val = self.y[int(0.6 * len(self.X)):int(0.8 * len(self.X))]
        # X_val_normalized = (X_val - mean) / std
        #
        # X_test = self.X[int(0.8 * len(self.X)):, :]
        # y_test = self.y[int(0.8 * len(self.X)):]
        # X_test_normalized = (X_test - mean) / std

        X_train = self.data['xtrain']
        X_train_normalized = self.data['xtrain']
        y_train = self.data['ytrain']
        X_val = self.data['xvalid']
        X_val_normalized = self.data['xvalid']
        y_val = self.data['yvalid']
        X_test = self.data['xvalid']
        X_test_normalized = self.data['xvalid']
        y_test = self.data['yvalid']

        # Set attribute values
        if (normalize):
            self.X_train = X_train_normalized
            self.y_train = y_train
            self.X_val = X_val_normalized
            self.y_val = y_val
            self.X_test = X_test_normalized
            self.y_test = y_test
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test

        # Confirm instances/features to be used for learning
        if (all_features):
            self.features = np.ones(X_train.shape[1])
        else:
            self.features = np.zeros(X_train.shape[1])
            while (np.sum(self.features) == 0):
                zero_p = random.uniform(0, 1)
                # zero_p = 0.5
                self.features = np.random.choice([0, 1], size=(X_train.shape[1],), p=[zero_p, (1 - zero_p)])
        self.features = list(np.where(self.features == 1)[0])

        if (all_instances):
            self.instances = np.ones(X_train.shape[0])
        else:
            self.instances = np.random.choice([0, 1], size=(X_train.shape[0],), p=[0.5, 0.5])
        self.instances = list(np.where(self.instances == 1)[0])
        # Train model and evaluate on validation/testing sets
        if (evaluate):
            self.fitClassifier()
            self.setValidationAccuracy()
            self.setTestAccuracy()
            self.setFun()

    def fitClassifier(self):
        self.clf = self.clf.fit(self.X_train[self.instances, :][:, self.features], self.y_train[self.instances])

    def setValidationAccuracy(self):
        y_pred = self.clf.predict(self.X_val[:, self.features])
        # print(confusion_matrix(self.y_val, y_pred))
        self.ValidationAccuracy = balanced_accuracy_score(self.y_val, y_pred)
        # X_New = np.zeros([1, np.size(self.X_train, 1)])
        # X_New[0, self.features] = 1
        # scores = 1-Fun(self.X_train, self.X_val, self.y_train, self.y_val, X_New, self.opts)
        # self.ValidationAccuracy = scores

    def setTestAccuracy(self):
        y_pred = self.clf.predict(self.X_test[:, self.features])
        # print('test', confusion_matrix(self.y_test, y_pred))
        # print(balanced_accuracy_score(self.y_test, y_pred))
        # print(accuracy_score(self.y_test, y_pred))
        # print(roc_auc_score(self.y_test, y_pred))
        self.TestAccuracy = balanced_accuracy_score(self.y_test, y_pred)
        # X_New = np.zeros([1, np.size(self.X_train, 1)])
        # X_New[0, self.features] = 1
        # scores = 1-Fun(self.X_train, self.X_val, self.y_train, self.y_val, X_New, self.opts)
        # self.TestAccuracy = scores

    def setCV(self):
        # X_New = np.zeros([1, np.size(self.X_train, 1)])
        # X_New[0, self.features] = 1
        # scores = 1-Fun(self.X_train, self.X_val, self.y_train, self.y_val, X_New, self.opts)
        # self.CV = scores
        scores = cross_val_score(self.clf, self.X_train[:][:, self.features], self.y_train[:], cv=self.folds,
                                 scoring='balanced_accuracy')
        # print(scores)
        # print(np.mean(scores))
        # print()
        self.CV = np.mean(scores)

    def setTrainSet(self, selected_instances):
        self.X_train = self.X_train[selected_instances]
        self.y_train = self.y_train[selected_instances]

    def setFeatures(self, selected_features):
        self.features = selected_features

    def setInstances(self, selected_instances):
        self.instances = selected_instances

    def getValidationAccuracy(self):
        return self.ValidationAccuracy

    def getTestAccuracy(self):
        return self.TestAccuracy

    def getCV(self):
        return self.CV

    def setFun(self):
        X_New = np.zeros([1, np.size(self.X_train, 1)])
        X_New[0, self.features] = 1
        scores = 1-Fun(self.X_train, self.X_val, self.y_train, self.y_val, X_New[0, :], self.opts)
        self.fun = scores
        self.curve.append(1-scores)

    def getFun(self):
        return self.fun



def fs(xtrain, xvalid, ytrain, yvalid, opts):
    label = -1
    header = None
    dataset = ProcessingDataset(xtrain, xvalid, ytrain, yvalid, opts, label, divide_dataset=False, header=header)
    if opts['classify'] == 'knn':  # K最近邻（KNN）算法
        # 在 KNN 中，para 表示选取最近的 para 个邻居来进行投票
        para = opts.get('knn_para', 5)
        classifier = KNeighborsClassifier(n_neighbors=para)
    elif opts['classify'] == 'svm':  # 支持向量机（SVM）算法
        # 在 SVM 中，para 不是算法的一部分，而是用于 RBF 核函数的参数，代表高斯核的宽度
        para = opts.get('svm_para', 1.0)
        classifier = SVC(kernel='rbf', C=para)
    elif opts['classify'] == 'rf':  # 随机森林（RF）算法
        # 在随机森林中，para 代表森林中树木的数量
        para = opts.get('rf_para', 100)
        classifier = RandomForestClassifier(n_estimators=para)
    elif opts['classify'] == 'dt':  # 决策树（Decision Tree）算法
        # 在决策树中，para 可以代表树的深度、节点最少样本数等参数
        para = opts.get('dt_para', None)
        classifier = DecisionTreeClassifier(max_depth=para)
    elif opts['classify'] == 'lr':  # 逻辑回归（Logistic Regression）算法
        # 在逻辑回归中，通常不涉及 para 参数
        classifier = LogisticRegression()
    dataset.divideDataset(classifier,
                          normalize=True,
                          shuffle=True,
                          all_features=True,
                          all_instances=True,
                          evaluate=True,
                          partial_sample=False,
                          folds=10)
    populationSize = opts['N']
    maxGenerations = opts['T']
    a = 16
    reductionRate = 0.5
    step = 10
    d = 10
    zeroP = 0.5
    alpha = 0.88,
    verbose = 0
    qualOnly = False
    timeout = np.inf
    noChange = np.inf
    evaluation = 'fun'
    # evaluation = 'validation'
    dim = np.size(xtrain, 1)
    c = []
    ###
    start = time.time()
    logDF = pd.DataFrame(columns=('generation', 'time', 'best_fitness', 'average_fitness', 'number_of_evaluations','best_solution', 'best_fitness_original', 'd', 'surrogate_level'))
    partialDataset = copy.copy(dataset)
    sampleSize = partialDataset.X_train.shape[0] // a
    randomSampling = random.sample(range(partialDataset.X_train.shape[0]), sampleSize)
    partialDataset.setInstances(randomSampling)
    if (verbose):
        print('Current Approx Sample Size:', len(partialDataset.instances))
        print('Current Population Size:', populationSize)
    task = 'feature_selection'
    indSize = partialDataset.X_train.shape[1]
    toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, partialDataset)
    if (alpha == 1):
        population = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, False)
    else:
        population = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, zeroP)

    bestTrueFitnessValue = -1 * np.inf
    sagaFeatureSubset = [1] * (len(partialDataset.features))
    qual = False

    numberOfEvaluations = 0
    generationCounter = 0
    maxAllowedSize = int(partialDataset.X_train.shape[0])

    d = indSize // 2
    surrogateLevel = 0
    startingPopulationSize = populationSize

    while True:
        # toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, partialDataset)
        log, population, d, c1 = EvolutionaryWrapperFeatureSelection.CHC(partialDataset,
                                                                     population,
                                                                     d=d,
                                                                     alpha=alpha,
                                                                     populationSize=populationSize,
                                                                     # maxGenerations=maxGenerations,
                                                                     maxGenerations=step,
                                                                     evaluation=evaluation,
                                                                     verbose=verbose,
                                                                     task=task)
        c = c + c1
        # d = log.iloc[-1]['d']
        generationCounter = generationCounter + step
        featureIndividual = log.iloc[-1]['best_solution']

        # Check if SAGA identified new feature subset

        if (sagaFeatureSubset != featureIndividual):
            trueBestInGeneration = np.round(
                100 * EvolutionaryWrapperFeatureSelection.evaluate(featureIndividual, 'feature_selection', evaluation,
                                                                   dataset, alpha=alpha)[0], 2)
            approxBestInGeneration = np.round(
                100 * EvolutionaryWrapperFeatureSelection.evaluate(featureIndividual, 'feature_selection', evaluation,
                                                                   partialDataset, alpha=alpha)[0], 2)
            numberOfEvaluations += 1
            end = time.time()
            row = [generationCounter, (end - start), approxBestInGeneration, 'NA',
                   numberOfEvaluations, featureIndividual, trueBestInGeneration, d, surrogateLevel]
            if (verbose):
                print(row)

            # Check if the original value improved
            if (trueBestInGeneration > bestTrueFitnessValue):
                bestTrueFitnessValue = trueBestInGeneration
                sagaFeatureSubset = featureIndividual
                sagaIndividual = tools.selBest(population, 1)
                if (verbose):
                    print('The best individual is saved', bestTrueFitnessValue)
                    print('Number of features in selected individual: ', np.sum(sagaFeatureSubset))
                row = [generationCounter, (end - start), approxBestInGeneration, 'NA',
                       numberOfEvaluations, sagaFeatureSubset, bestTrueFitnessValue, d, surrogateLevel]
                logDF.loc[len(logDF)] = row

            # A possible false optimum is detected.

            elif (len(partialDataset.instances) < maxAllowedSize):
                if (verbose):
                    print('A possible false optimum is detected!')
                best_approx_fitness_value = 0
                a = a / 2
                sampleSize = int(partialDataset.X_train.shape[0] / a)
                populationSize = int(populationSize * reductionRate)
                surrogateLevel += 1
                d = indSize // 2
                onesP = (np.sum(sagaIndividual) / indSize)
                if (sampleSize < maxAllowedSize):
                    randomSampling = random.sample(range(partialDataset.X_train.shape[0]), sampleSize)
                    partialDataset.setInstances(randomSampling)
                    if (alpha == 1):
                        newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize,
                                                                                      False)
                    else:
                        newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize,
                                                                                      1 - onesP)

                    population[:] = tools.selBest(sagaIndividual + newInd, populationSize)
                    if (verbose):
                        print('Current Approx Sample Size:', len(partialDataset.instances))
                        print('Current Population Size:', populationSize)

                else:
                    if (qualOnly):
                        end = time.time()
                        qualTime = (end - start)
                        return logDF
                    partialDataset = copy.copy(dataset)
                    onesP = (np.sum(sagaIndividual) / indSize)
                    populationSize = 40
                    newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize, 1 - onesP)
                    population[:] = tools.selBest(sagaIndividual + newInd, populationSize)

                    sagaFeatureSubset = copy.copy(dataset)
                    toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation,
                                                                                sagaFeatureSubset)
                    if (verbose):
                        print('Approximation stage is over!')
                        print('Current Approx Sample Size:', len(partialDataset.instances))
                        print('Current Population Size:', populationSize)

                    end = time.time()
                    qualTime = (end - start)
                    log, population, d, c2 = EvolutionaryWrapperFeatureSelection.CHC(dataset,
                                                                                 population,
                                                                                 populationSize=populationSize,
                                                                                 alpha=alpha,
                                                                                 maxGenerations=maxGenerations,
                                                                                 maxNochange=noChange,
                                                                                 timeout=timeout - qualTime,
                                                                                 evaluation=evaluation,
                                                                                 verbose=verbose,
                                                                                 task=task)
                    c = c + c2
                    break

            elif (len(partialDataset.instances) >= maxAllowedSize):
                break

        # The current approximation converged!
        else:

            # Check if the current appoximation is the maximum allowed.

            if (len(partialDataset.instances) >= maxAllowedSize):
                break

            if (verbose):
                print('The approximation converged!')
            best_approx_fitness_value = 0
            a = a / 2
            sampleSize = int(partialDataset.X_train.shape[0] / a)
            populationSize = int(populationSize * reductionRate)
            surrogateLevel += 1
            d = indSize // 2
            onesP = (np.sum(sagaIndividual) / indSize)
            if (sampleSize < maxAllowedSize):
                randomSampling = random.sample(range(partialDataset.X_train.shape[0]), sampleSize)
                partialDataset.setInstances(randomSampling)
                if (alpha == 1):
                    newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize,
                                                                                  False)
                else:
                    newInd = EvolutionaryWrapperFeatureSelection.createPopulation(populationSize, indSize,
                                                                                  1 - onesP)

                population[:] = tools.selBest(sagaIndividual + newInd, populationSize)
                if (verbose):
                    print('Current Approx Sample Size:', len(partialDataset.instances))
                    print('Current Population Size:', populationSize)

            else:
                if (qualOnly):
                    end = time.time()
                    qualTime = (end - start)
                    return logDF
                partialDataset = copy.copy(dataset)
                onesP = (np.sum(sagaIndividual) / indSize)
                populationSize = 40
                newInd = EvolutionaryWrapperFeatureSelection.createPopulation(startingPopulationSize, indSize,
                                                                              1 - onesP)
                population[:] = tools.selBest(sagaIndividual + newInd, startingPopulationSize)
                partialDataset = copy.copy(dataset)
                toolbox = EvolutionaryWrapperFeatureSelection.createToolbox(indSize, task, evaluation, partialDataset)
                if (verbose):
                    print('Approximation stage is over!')
                    print('Current Approx Sample Size:', len(partialDataset.instances))
                    print('Current Population Size:', startingPopulationSize)
                end = time.time()
                qualTime = (end - start)

                log, population, d, c3 = EvolutionaryWrapperFeatureSelection.CHC(dataset,
                                                                             population,
                                                                             populationSize=startingPopulationSize,
                                                                             alpha=alpha,
                                                                             maxGenerations=maxGenerations,
                                                                             maxNochange=noChange,
                                                                             timeout=timeout - qualTime,
                                                                             verbose=verbose,
                                                                             task=task)
                c = c + c3
                break
    # print(qualTime)
    for index, row in log.iterrows():
        log.at[index, 'time'] = log.loc[index, 'time'] + qualTime
        log.at[index, 'number_of_evaluations'] = log.loc[index, 'number_of_evaluations'] + numberOfEvaluations
        log.at[index, 'generation'] = log.loc[index, 'generation'] + logDF.iloc[-1]['generation']
    log['best_fitness_original'] = 100 * log['best_fitness']
    log['best_fitness'] = 100 * log['best_fitness']

    logDF = pd.concat((logDF, log))

    feature_subset = logDF.iloc[-1]['best_solution']
    feature_subset = np.array(feature_subset)
    accuracy = np.round(100 * EvolutionaryWrapperFeatureSelection.evaluate(feature_subset, 'feature_selection', 'test', dataset, 1)[0],2)
    # Best feature subset
    Gbin = feature_subset
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # curve = np.zeros([0, opts['T']])
    c = np.array(c)
    repeated_c = np.repeat(c, 5, axis=0)  # 将 c 重复为 (5x, 1) 形状的数组
    c = 1 - repeated_c  # 执行减法操作
    sorted_idx = np.argsort(c)[::-1]  # 对 c 中的元素进行排序，并返回排序后的索引
    sorted_c = c[sorted_idx]  # 根据排序后的索引，获取对应的数值
    num_points = opts['T']
    indices = np.round(np.linspace(0, np.size(sorted_c, 0) - 1, num_points)).astype(int)
    curve = sorted_c[indices]
    curve = curve.reshape(1, -1)
    saga_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}

    return saga_data