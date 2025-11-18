import sklearn as sk
import numpy as np
import random
import csv
import time
from copy import copy
from sklearn.neighbors import KNeighborsClassifier
from Function import Fun


#%% defining fitness function and particle class
def fitness(x,data,class_identifier,percent_training=70,repeat=10):
    # importing data
    # with open(datafile, newline='') as csvfile: # load as string to check for classifiers
    #     data_raw = list(csv.reader(csvfile))
    # ncol_data = len(data_raw[0])
    # nrow_data = len(data_raw)
    # target = []
    # for i in range(nrow_data):
    #     target.append(data_raw[i][class_identifier])
    #
    # feature_columns = np.array(range(ncol_data))
    # feature_columns = np.delete(feature_columns,class_identifier,0)
    # data = np.loadtxt(open(datafile, "rb"), delimiter=",",usecols=feature_columns) # load as numeric, rb-read binary
    #
    # nfeature = data.shape[1]
    # ndata = data.shape[0]
    # ntraining = round(ndata*percent_training/100)
    # ntest = ndata-ntraining
    xtrain = data['xtrain']
    xvalid = data['xvalid']
    ytrain = data['ytrain']
    yvalid = data['yvalid']
    opts = data['opts']
    result = data['result']
    active_feature = np.array([i for i, y in enumerate(sigmoid(x)>=0.5) if y])

    if(len(active_feature)==0):
        fit = 1
    else:
        X_New = np.zeros([1, np.size(xtrain, 1)])
        X_New[0, active_feature] = 1
        fit = Fun(xtrain, xvalid, ytrain, yvalid, X_New[0, :], opts)
        result['Gbin'].append(X_New[0, :])
        result['curve'].append(fit)

    #print("active")
    #print(active_feature)
    # define training data and target
    # fitness = []
    # for j in range(repeat):
    #     # define training data and target
    #     training_index = random.sample(range(ndata),ntraining)
    #     training_index = np.array(training_index)
    #     training_data = data[training_index, : ]
    #     training_data = training_data[:,active_feature] # remove class identifier
    #     training_target = []
    #     for i in training_index:
    #         training_target.append(target[i])
    #
    #     # define test data and target
    #     test_index =  list(set(range(ndata)).symmetric_difference(set(training_index)))
    #     test_index = np.array(test_index) # convert it for easy access
    #     test_data = data[test_index,:]
    #     test_data = test_data[:,active_feature] # get active features only
    #     test_target = []
    #     for i in test_index:
    #         test_target.append(target[i])
    #
    #     feature_test = KNeighborsClassifier(n_neighbors=5)
    #     feature_test.fit(training_data,training_target)
    #     fitness.append (1.0-feature_test.score(test_data,test_target))
    # mean_fitness = np.average(fitness)
    return fit

def update_position(particle,data,class_identifier,percent_training=70,repeat=10):
    particle.x = particle.x + particle.v
    particle.fitness = fitness(particle.x,data,class_identifier,percent_training,repeat)
    return particle
    
def update_velocity(particle,gbest,vmax = 6):
    W = 0.7298
    c1 = 1.149618
    c2 = 1.149618
    particle.v = ( W*particle.v + # inertia term
        c1*random.random()*(particle.pbest-particle.x) + # cognitive term
        c2*random.random()*(gbest-particle.x) # social term
        )
    for i in range(particle.v.shape[0]):
        if abs(particle.v[i])>vmax:
            particle.v[i] = vmax*particle.v[i]/abs(particle.v[i])
    return particle

def update_pbest(particle,PG1rule = True):
    nactive = 0
    nactive_pbest = 0
    nfeature = particle.x.shape[0]
    for i in range(nfeature):  # count active features in x and pbest
        if sigmoid(particle.x[i])>=0.5: # 0.5 is the threshold for active feature
            nactive = nactive+1
        if sigmoid(particle.pbest[i])>=0.5:
            nactive_pbest = nactive_pbest + 1
    if PG1rule:
        if particle.fitness < particle.pbest_fit and nactive <= nactive_pbest: # line 6 of algorithm --- 
            particle.pbest = particle.x
            particle.pbest_fit = particle.fitness
        elif 0.95*particle.fitness < particle.pbest_fit and nactive < nactive_pbest: # line 9
            particle.pbest = particle.x
            particle.pbest_fit = particle.fitness
    else:
        if particle.fitness < particle.pbest_fit: 
            particle.pbest = particle.x
            particle.pbest_fit = particle.fitness
    return particle

def update_gbest(pop,gbest,gbest_fit,PG1rule=True):
    pop_size = len(pop)
    nfeature = pop[0].x.shape[0]
    for i in range(pop_size):
        nactive_pbest = 0
        nactive_gbest = 0
        for j in range(nfeature):  # count active features in pbest and gbest
            if sigmoid(pop[i].pbest[j])>=0.5: # 0.5 is the threshold for active feature
                nactive_pbest = nactive_pbest+1
            if sigmoid(gbest[j])>=0.5:
                nactive_gbest = nactive_gbest + 1
        if PG1rule:
            if pop[i].pbest_fit < gbest_fit and nactive_pbest <= nactive_gbest: # line 12-14
                #print("update gbest")
                gbest = pop[i].pbest
                gbest_fit = pop[i].pbest_fit
            elif 0.95*pop[i].pbest_fit < gbest_fit and nactive_pbest < nactive_gbest: # line 15-17
                #print("update gbest")
                gbest = pop[i].pbest
                gbest_fit = pop[i].pbest_fit
        else:
            if pop[i].pbest_fit < gbest_fit: # line 12-14
                #print("update gbest")
                gbest = pop[i].pbest
                gbest_fit = pop[i].pbest_fit
    return gbest,gbest_fit

class Particle:
    def __init__(self, variable_count,active_feature_count,fitness_function,datafile,class_identifier,percent_training=70,repeat=10):
        self.variable_count = variable_count
        #self.x = 0.6*np.ones(self.variable_count) # particle position
        self.x = -np.ones(self.variable_count) # particle position
        feature_id_to_activate = random.sample(range(variable_count),active_feature_count)
        #self.x[feature_id_to_activate] = 0.6+random.random()*0.4
        self.x[feature_id_to_activate] = 1
        vlist = []
        for i in range(variable_count):
            vlist.append( (random.random()-  self.x[i])/2 ) # see eq. 2
        self.v = np.array(vlist) # particle velocity
        self.pbest = self.x # particle best position
        self.fitness = fitness_function(self.x,datafile,class_identifier,percent_training,repeat)
        self.pbest_fit = self.fitness

#%% initialize population following mixed/hybrid rule
def initialize_pop(population_size,variable_count,small_feature_percent,class_identifier,data, percent_training=70,repeat=10):
    pop = []
    n_small = round(small_feature_percent/100*population_size) # number of individual with low feature active
    n_large = population_size-n_small # number of individual with many features
    n_active_small = round(0.12 * variable_count) # number of active feature for small individual
    for i in range(n_small):
        pop.append(Particle(variable_count,n_active_small,fitness,data,class_identifier,percent_training,repeat))  # add small individual to the population
    minimum_feature_large = round(variable_count/2) # minimum number of active feature for large individual
    for i in range(n_large):
        n_active = random.randint(minimum_feature_large,variable_count)
        pop.append(Particle(variable_count,n_active,fitness,data,class_identifier,percent_training,repeat))  # add large individual to the population
    return pop
    
#when using random initialization
def initialize_pop_fullrandom(population_size,variable_count,small_feature_percent,class_identifier,data,percent_training=70,repeat=10):
    pop = []
    for i in range(population_size):
        n_active = random.randint(1,variable_count) # zero active feature is not feasible
        pop.append(Particle(variable_count,n_active,fitness,data,class_identifier,percent_training,repeat))  # add individual with random number of active feature to the population
    return pop


def get_pop_fitness(pop):
    fitness_list = np.zeros(len(pop))
    for i in range(len(pop)):
        fitness_list[i]= pop[i].fitness
    return fitness_list

def sigmoid(x):
    return 1/(1+np.exp(-2*x))

#%% 
def runPSO(number_of_iteration, p, data):
    usePG1rule=True
    p_fitness = get_pop_fitness(p)  # example on how to evaluate the whole population at once
    # print("PSO started")
    gbest_id = p_fitness.argmax()
    gbest = p[gbest_id].x # gbest only store position
    gbest_fit = max(p_fitness)
    SOPF_gbest_fit = gbest_fit # check this later to trigger SOPF
    nparticle = p_fitness.shape[0]
    class_identifier = 0
    percent_training = 70
    percent_of_small_feature = 30  # percent of initial population with small feature
    dim = np.size(data['xtrain'], 1)
    number_of_feature = dim  # 13 for wine, 34 for ionosphere,...
    repeat_crosscorr =10 # easy to change the number of repetition for crosscorrelation
    embedSOPF = True  # True => use SOPF inside PSO, only on gbest
    triggerSOPF_at = 10
    for i in range(number_of_iteration):
        # print("iteration number"+str(i))
        for j in range(nparticle):
            p[j]=update_pbest(p[j],usePG1rule)
            gbest,gbest_fit=update_gbest(p,gbest,gbest_fit,usePG1rule)
        for j in range(nparticle):
            p[j]=update_velocity(p[j],gbest)
            p[j]=update_position(p[j],data,class_identifier,percent_training,repeat_crosscorr)
        if embedSOPF: # check whether embedsopf is going to be used
            if i%triggerSOPF_at==0: # check that it is at the correct number of iteration to do SOPF
                # print("Trigger SOPF at iteration number"+str(i))
                if SOPF_gbest_fit<=gbest_fit:   # check that there is no improvement since the last "triggerSOPF_at" iterations
                    # print("No improvement detected, starting SOPF")
                    for i in range(number_of_feature):
                        temp_gbest = copy(gbest)
                        if temp_gbest[i]<=0:  # sigmoid at 0.5
                            temp_gbest[i]=1 # activate
                        else:
                            temp_gbest[i]=-1 # deactivate
                        temp_fitness = fitness(np.array(temp_gbest),data,
                        class_identifier, percent_training,repeat_crosscorr)
                        if(temp_fitness<gbest_fit):
                            gbest = temp_gbest
                            gbest_fit = temp_fitness
                            SOPF_gbest_fit = gbest_fit
                            break
    return gbest,gbest_fit


def fs(xtrain, xvalid, ytrain, yvalid, opts):
    result = {'Gbin': [], 'curve': []}
    data = {'xtrain': xtrain, 'xvalid': xvalid, 'ytrain': ytrain, 'yvalid': yvalid, 'opts': opts, 'result': result}
    class_identifier = 0
    percent_training = 70
    number_of_iteration = opts['T'] # PSO iterations
    percent_of_small_feature = 30  # percent of initial population with small feature
    population_size = opts['N']
    dim = np.size(xtrain, 1)
    number_of_feature = dim  # 13 for wine, 34 for ionosphere,...
    repeat_crosscorr =10 # easy to change the number of repetition for crosscorrelation
    SOPF = True # True => use SOPF after PSO
    # %% test this
    p = initialize_pop(population_size, number_of_feature, percent_of_small_feature, class_identifier, data, percent_training,repeat_crosscorr)  # input: pop size, n_feature, percent of population with small feature, classifier index
    gbest,gbest_fit = runPSO(number_of_iteration, p,data)
    # print("Error rate:")
    # print(gbest_fit)
    # print("Accuracy:")
    # print(1-gbest_fit)
    nfeature = gbest.shape[0]
    nactive_gbest = 0
    active_feature = []
    for j in range(nfeature):  # count active features in pbest and gbest
        if sigmoid(gbest[j])>=0.5:
            nactive_gbest = nactive_gbest + 1
            active_feature.append(1)
        else:
            active_feature.append(0)
    # print("Active features")
    # print(active_feature)
    # print("Number of active features:")
    # print(nactive_gbest)
    # print ("time for main PSO: %f seconds" , time_end_mainPSO-time_start)

    # when using SOPF at the end of maximum set iteration
    if SOPF:
        for i in range(number_of_feature):
            temp_active = copy(active_feature)
            temp_nactive = copy(nactive_gbest)
            if active_feature[i]==0:
                temp_active[i]=1
                temp_nactive = temp_nactive+1
            else:
                temp_active[i]=0
                temp_nactive = temp_nactive-1
            temp_fitness = fitness(np.array(temp_active),data,
            class_identifier, percent_training,repeat_crosscorr)
            if(temp_fitness<gbest_fit):
                active_feature = temp_active
                gbest_fit = temp_fitness
                nactive_gbest = temp_nactive
                break
        # print("After SOPF:")
        # print("Error rate:")
        # print(gbest_fit)
        # print("Accuracy:")
        # print(1-gbest_fit)
        # print("Active features")
        # print(active_feature)
        # print("Number of active features:")
        # print(nactive_gbest)
        # time_end_withSOPF  = time.time()
        # print ("time after SOPF: %f seconds " ,  time_end_withSOPF-time_start)
    result = data['result']
    min_index = result['curve'].index(min(result['curve']))
    Gbin = result['Gbin'][min_index]
    num_feat = np.sum(Gbin)
    c = np.array(result['curve'])
    sorted_idx = np.argsort(c)[::-1]  # 对 c 中的元素进行排序，并返回排序后的索引
    sorted_c = c[sorted_idx]  # 根据排序后的索引，获取对应的数值
    num_points = opts['T']
    indices = np.round(np.linspace(0, np.size(sorted_c, 0) - 1, num_points)).astype(int)
    curve = sorted_c[indices]
    curve = curve.reshape(1, -1)
    PSOPG_SOPF_data = {'sf': Gbin, 'c': curve, 'nf': num_feat}

    return PSOPG_SOPF_data