# SpFSR: Feature Selection and Ranking via SPSA-BB
# Required Python version >= 3.6
# D. Akman & Z. D. Yenice
# GPL-3.0, 2022
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589


import numpy as np
from spfsr2 import SpFSR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def fs(xtrain, xvalid, ytrain, yvalid, opts):
    # 拼接 x 矩阵
    x = np.concatenate((xtrain, xvalid), axis=0)
    # 拼接 y 矩阵
    y = np.concatenate((ytrain, yvalid), axis=0)
    dim = np.size(xtrain, 1)
    pred_type = 'c'
    scoring = 'accuracy'
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
    wrapper = classifier

    # set the engine parameters
    sp_engine = SpFSR(xtrain, ytrain, pred_type=pred_type, scoring=scoring, wrapper=wrapper)

    # run the spFSR engine
    # available engine parameters (with default values in parentheses):
    # 1.  num_features (0): number of features to select - a value of 0 results in automatic feature selection
    # 2.  iter_max (100): max number of iterations
    # 3.  stall_limit (35): when to restart the search (up to iter_max) - should be about iter_max/3
    # 4.  n_samples_max (2500): max number of randomly selected observations to be used during search
    #     can be 'None' for using all available observations
    # 5.  ft_weighting (False): if features should be weighted by their importance values - this usually helps with kNN's.
    # 6.  use_hot_start (True): if hot start is to be used where initial feature importance vector is
    #     determined by random forest importance (RFI) - this usually results in faster convergence
    # 7.  hot_start_range (0.2): range for the initial feature importance vector in case of a hot start
    #     for example: for a range of 0.2, most important RFI feature will have an imp. value of 0.1 and
    #     the least important will have an imp. value of -0.1 - a value of 0 is also possible and
    #     it will result in all RFI-selected features to have 0 imp. values
    # 8.  rf_n_estimators_hotstart (50): number of estimators for hot start RFI
    # 9.  rf_n_estimators_filter (5): number of estimators for prediction RFI in FILTER mode
    # 10.  gain_type ('bb'): either 'bb' (Barzilai & Borwein) gains or 'mon' (monotone) gains as the step size during search
    # 11.  cv_folds (5): number of folds to use during (perhaps repeated) CV both for evaluation and gradient evaluation
    # 12.  num_grad_avg (4): number of gradient estimates to be averaged for determining search direction
    #     for better gradient estimation, try increasing this number - though this will slow down the search
    # 13. cv_reps_eval (3): number of CV repetitions for evaluating a candidate feature set
    # 14. cv_reps_grad (1): number of CV repetitions for evaluating y-plus and y-minus solutions during gradient estimation
    # 15. stall_tolerance (1e-8): tolerance in objective function change for stalling
    # 16. display_rounding (3): number of digits to display during algorithm execution
    # 17. is_debug (False): whether detailed search info should be displayed for each iteration
    # 18. random_state(1): seed for controlling randomness in the execution of the algorithm
    # 19. n_jobs (1): number of cores to be used in CV - this will be passed into cross_val_score()
    # 20. print_freq (10): iteration print frequency for the algorithm output
    sp_run = sp_engine.run(num_features=0, iter_max=opts['T'], cv_folds=5, display_rounding=4, xtrain=xtrain, ytrain=ytrain, xvalid=xvalid, yvalid=yvalid, opts=opts)

    # get the results of the run
    sp_results = sp_run.results
    Gbin = np.zeros((1, dim))
    curve = np.zeros([1, opts['T']], dtype='float')
    sel_index = sp_results['selected_features']
    Gbin[0, sel_index] = 1
    num_feat = sp_results['selected_num_features']
    iter_results = sp_results['iter_results']['values']
    best = 1-iter_results[0]
    for i, value in enumerate(iter_results[-opts['T']:]):
        value = 1-value
        if value<best:
            best = value
        curve[0, i] = best
    spfsr_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}
    return spfsr_data