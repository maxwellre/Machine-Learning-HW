# Homework 4: Classification using SVM and Adaboost
# Created on 05/16/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import contextlib
import numpy as np
import numpy.matlib as matlib
import numpy.linalg as nla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC

#-----------------------------------------------------------------------------------------------------------------------
# Configurations
dimen = 2 # data dimension

np.random.seed(0)

#-----------------------------------------------------------------------------------------------------------------------
# Functions
def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    Cov = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        Cov = Cov +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, Cov, data_num)
    return ret_data, Cov

# Evaluate binary classification result
def evalBCResult(pLabel, tLabel):
    incorrInd = np.squeeze(np.asarray((pLabel.flatten() != tLabel)))
    class0Ind = (tLabel == 0)
    class1Ind = (tLabel == 1)
    incorrPr0 = np.sum(incorrInd[class0Ind])/(class0Ind.shape[0])
    incorrPr1 = np.sum(incorrInd[class1Ind])/(class1Ind.shape[0])
    print("Class 0 - error = {0:4.1f}%\nClass 1 - error = {1:4.1f}%\n".format(100*incorrPr0,100*incorrPr1))
    return incorrInd

# Visualize data
def disp2DResult(data_in, one_hot, disp_now=1, label_ext = ''):
    label_num = one_hot.shape[1]
    for i in range(label_num):
        plt.scatter(data_in[one_hot[:,i],0],data_in[one_hot[:,i],1],s=5,label="Class {:2d} {:s}".format(i+1,label_ext))
    plt.xlabel('Dimension 0',fontsize=10)
    plt.ylabel('Dimension 1',fontsize=10)
    plt_ax = plt.gca()
    plt_ax.set_aspect('equal', 'box')
    if disp_now:
        plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Generating data
data_num = 300

# Class 0
theta0 = 0
m0 = np.zeros(dimen)
lamb0 = [2,1]
U0 = np.mat([[np.cos(theta0), np.sin(theta0)],[-np.sin(theta0), np.cos(theta0)]]).T

x0, C0 = gaussianFromEigen(m0, lamb0, U0, data_num)

# Class 1
thetaA = -3*(np.pi)/4
pi_A = 1/3
mA = np.array([-2,1])
lambA = [2,1/4]
UA = np.mat([[np.cos(thetaA), np.sin(thetaA)],[-np.sin(thetaA), np.cos(thetaA)]]).T

thetaB = (np.pi)/4
pi_B = 2/3
mB = np.array([3,2])
lambB = [3,1]
UB = np.mat([[np.cos(thetaB), np.sin(thetaB)],[-np.sin(thetaB), np.cos(thetaB)]]).T

x1A, C1A = gaussianFromEigen(mA, lambA, UA, data_num)
x1B, C1B = gaussianFromEigen(mB, lambB, UB, data_num)
mixGaussPi = np.random.uniform(0.0,1.0,data_num)

x1 = np.concatenate((x1A[mixGaussPi <= pi_A,:],x1B[mixGaussPi > pi_A,:]),axis=0)
x1 = x1[np.random.permutation(data_num),:] # Reshuffle the gaussian mixture data

print('C0 =\n',C0, '\n C1A =\n',C1A,'\n C1B =\n',C1B)

train_num = int(data_num*0.6666)

xTrain = np.concatenate((x0[:train_num,:],x1[:train_num,:]),axis=0) # data
tTrain = np.concatenate((np.zeros(train_num),np.ones(train_num))) # label

xTest = np.concatenate((x0[train_num:,:],x1[train_num:,:]),axis=0) # data
tTest = np.concatenate((np.zeros(data_num-train_num),np.ones(data_num-train_num))) # label

# Shuffle the training set
trainLen = xTrain.shape[0]
randInd = np.random.permutation(trainLen)
xTrain = xTrain[randInd,:]
tTrain = tTrain[randInd]

#-----------------------------------------------------------------------------------------------------------------------
# 1) SVM-SMO
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")


l = [0.1, 2.0, 10.0] # kernel radius

for i in range(3):
    gamma = 0.5/(l[i]**2)
    clf = SVC(kernel='rbf', gamma=gamma)
    clf.fit(xTrain, tTrain)
    support_vect = clf.support_vectors_
    vect_num = clf.n_support_
    support_vect_ratio = 0.5*sum(vect_num)/train_num

    tPred = clf.predict(xTest)
    evalBCResult(tPred, tTest)

    # Visualize data and classification error (and decision boundary)
    plt.subplot2grid((1, 3), (0, i), rowspan=1, colspan=1)

    plt.scatter(support_vect[:,0],support_vect[:,1],c='k',s=20,label="Support Vectors",facecolors=None)

    disp2DResult(xTrain, np.column_stack((1 - tTrain, tTrain)).astype(bool),0,label_ext='Train')
    disp2DResult(xTest, np.column_stack((1 - tTest, tTest)).astype(bool),0,label_ext='Test')

    xRange = np.arange(min(np.min(xTrain[:,0]),np.min(xTest[:,0])),max(np.max(xTrain[:,0]),np.max(xTest[:,0])),0.05)
    yRange = np.arange(min(np.min(xTrain[:,1]),np.min(xTest[:,1])),max(np.max(xTrain[:,1]),np.max(xTest[:,1])),0.05)
    xGrid, yGrid = np.meshgrid(xRange, yRange, sparse=False, indexing='xy')
    xGrid = np.reshape(xGrid, (xGrid.size,1))
    yGrid = np.reshape(yGrid, (yGrid.size,1))
    deciBoundX = np.column_stack((xGrid,yGrid))
    classPr = clf.predict(deciBoundX)
    plt.scatter(deciBoundX[(classPr == 1),0],deciBoundX[(classPr == 1),1],s=0.01,c='g',label='Class 1 Boundary')
    if (i ==1):
        (plt.gca()).legend(loc='upper left',bbox_to_anchor=(0.25, -0.15))
    plt.title("l={0:.1f}, fraction of support vectors = {1:.1f}%" \
     .format(l[i],100*support_vect_ratio),fontsize=12)

plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# 2)