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
# Class
# Implementation of my decision stumps classifier
class DecisionStumps:
    def __init__(self, weight):
        self.d = None
        self.slct_dimen = None
        self.data_min = None
        self.data_max = None
        self.fit_err = None
        self.decision_bound = None
        self.pred_sign = None
        self.weight = weight

    def fit(self, data_in, labels, used_bound_ind, scan_num):
        [dataLen, self.d] = np.shape(data_in)
        fit_err = np.empty((self.d,scan_num))

        self.data_min = np.zeros(self.d)
        self.data_max = np.ones(self.d)

        pred_sign_flip = np.zeros((self.d,scan_num),dtype=bool)
        err_origin = np.empty((self.d,scan_num))
        err_flip = np.empty((self.d, scan_num))

        for i in range(self.d):
            self.data_min[i] = np.min(data_in[:,i])
            self.data_max[i] = np.max(data_in[:,i])
            step_size = (self.data_max[i]-self.data_min[i])/scan_num
            for j in range(scan_num):
                decision_bound = self.data_min[i] + j*step_size
                pred_label = np.zeros(dataLen)
                pred_label[data_in[:,i] > decision_bound] = 1

                err_ind = (pred_label != labels)
                err_origin[i, j] = np.sum(self.weight[err_ind])
                err_flip[i, j] = np.sum(self.weight[np.logical_not(err_ind)])

                if (err_origin[i, j] > err_flip[i, j]):
                    pred_sign_flip[i, j] = True
                    fit_err[i, j] = err_flip[i, j]
                else:
                    fit_err[i, j] = err_origin[i, j]

        fit_err[used_bound_ind] = 1
        optimal_ind = np.unravel_index(fit_err.argmin(), fit_err.shape)
        used_bound_ind[optimal_ind] = True

        self.pred_sign_flip = pred_sign_flip[optimal_ind]
        self.slct_dimen = optimal_ind[0]
        self.decision_bound = self.data_min[self.slct_dimen] + optimal_ind[1] * step_size

        # pred_label = np.zeros(dataLen)
        # class1_pred_ind = np.logical_xor((data_in[:, self.slct_dimen] > self.decision_bound), self.pred_sign_flip)
        # pred_label[class1_pred_ind] = 1
        # fit_err_ind = np.not_equal(pred_label, labels)
        #
        # print("Err rate = {:.1f}%", 100*np.sum(fit_err_ind) / dataLen)
        # print("Sign flipped = ",self.pred_sign_flip)
        #
        # plt.figure()
        # disp2DResult(xTrain, np.column_stack((1 - tTrain, tTrain)).astype(bool), 0)
        # self.plotBound()
        # plt.plot(data_in[fit_err_ind, 0], data_in[fit_err_ind, 1], 'k.', markersize=2)
        # plt.show()

        return used_bound_ind

    def predict(self,data_in):
        [dataLen, d] = np.shape(data_in)
        if (self.d != d):
            raise Exception('Dimension mismatch!')
        pred_label = np.zeros(dataLen)
        class1_pred_ind = np.logical_xor((data_in[:,self.slct_dimen] > self.decision_bound), self.pred_sign_flip)
        pred_label[class1_pred_ind] = 1
        return pred_label

    def plotBound(self):
        if (self.d == 2):
            if (self.slct_dimen == 1):
                plt.plot([self.data_min[0], self.data_max[0]], [self.decision_bound, self.decision_bound],'k',
                         linewidth=1)
            elif (self.slct_dimen == 0):
                plt.plot([self.decision_bound, self.decision_bound], [self.data_min[1], self.data_max[1]],'k',
                         linewidth=1)
        else:
            raise Exception('Can display 2D boundary only!')

# Implementation of my Adaboost
class Adaboost:
    def __init__(self, M = 400, weight = None):
        self.M = M
        self.weight = weight
        self.clfs = []
        self.errs = np.ones(self.M)
        self.alphas = np.ones(self.M)

    def fit(self, data_in, labels, dispBound = False, scan_num = 200):
        [dataLen, self.d] = np.shape(data_in)
        if (self.weight is None):
            self.weight = np.ones(dataLen) / dataLen

            used_bound_ind = np.zeros((self.d,scan_num),dtype=bool)
        for i in range(self.M):
            clf = DecisionStumps(self.weight)
            used_bound_ind = clf.fit(data_in, labels, used_bound_ind, scan_num)

            # fit_err_ind = clf.fastfit(data_in, labels)

            pred_label = clf.predict(data_in)
            fit_err_ind = np.not_equal(pred_label,labels)

            # if dispBound:
            #     clf.plotBound()

            # plt.show()

            self.errs[i] = np.sum(self.weight[fit_err_ind])/np.sum(self.weight)

            self.alphas[i] = np.log((1-self.errs[i])/(self.errs[i] + 1e-16))

            # temp = np.ones(dataLen)
            # temp[fit_err_ind] = -1
            # self.weight *= np.exp(self.alphas[i] * temp)

            self.weight *= np.exp(self.alphas[i]*fit_err_ind)

            self.weight = self.weight / np.sum(self.weight)

            self.clfs.append(clf)

        print("Fit errs = ",self.errs)
        print("Fit alphas = ", self.alphas)
        print("Final used_bound_ind = ",used_bound_ind)

    def predict(self, data_in):
        dataLen = data_in.shape[0]
        pred_label = np.zeros(dataLen)
        pred_all = np.zeros(dataLen)
        for i in range(self.M):
            curr_pred = self.clfs[i].predict(data_in)
            curr_pred[curr_pred == 0] = -1
            pred_all += ( curr_pred * self.alphas[i] )
        pred_label[pred_all > 0] = 1
        return pred_label

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
mA = np.array([-2,1])*2
lambA = [2,1/4]
UA = np.mat([[np.cos(thetaA), np.sin(thetaA)],[-np.sin(thetaA), np.cos(thetaA)]]).T

thetaB = (np.pi)/4
pi_B = 2/3
mB = np.array([3,2])*2
lambB = [3,1]
UB = np.mat([[np.cos(thetaB), np.sin(thetaB)],[-np.sin(thetaB), np.cos(thetaB)]]).T

x1A, C1A = gaussianFromEigen(mA, lambA, UA, data_num)
x1B, C1B = gaussianFromEigen(mB, lambB, UB, data_num)
mixGaussPi = np.random.uniform(0.0,1.0,data_num)

x1 = np.concatenate((x1A[mixGaussPi <= pi_A,:],x1B[mixGaussPi > pi_A,:]),axis=0)
x1 = x1[np.random.permutation(data_num),:] # Reshuffle the gaussian mixture data

print('C0 =\n',C0, '\n C1A =\n',C1A,'\n C1B =\n',C1B)

train_num = int(data_num*0.6667)

xTrain = np.concatenate((x0[:train_num,:],x1[:train_num,:]),axis=0) # data
tTrain = np.concatenate((np.zeros(train_num),np.ones(train_num))) # label

xTest = np.concatenate((x0[train_num:,:],x1[train_num:,:]),axis=0) # data
tTest = np.concatenate((np.zeros(data_num-train_num),np.ones(data_num-train_num))) # label

# Shuffle the training set
trainLen = xTrain.shape[0]
randInd = np.random.permutation(trainLen)
xTrain = xTrain[randInd,:]
tTrain = tTrain[randInd]
# xTrain = xTrain[randInd[:20],:]
# tTrain = tTrain[randInd[:20]]

xMin = min(np.min(xTrain[:,0]),np.min(xTest[:,0]))
xMax = max(np.max(xTrain[:,0]),np.max(xTest[:,0]))
yMin = min(np.min(xTrain[:,1]),np.min(xTest[:,1]))
yMax = max(np.max(xTrain[:,1]),np.max(xTest[:,1]))
#-----------------------------------------------------------------------------------------------------------------------
# 1) SVM-SMO
# l = [0.1, 2.0, 10.0] # kernel radius
# plt.figure()
# plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
# for i in range(3):
#     gamma = 0.5/(l[i]**2)
#     clf = SVC(kernel='rbf', gamma=gamma)
#     clf.fit(xTrain, tTrain)
#     support_vect = clf.support_vectors_
#     vect_num = clf.n_support_
#     support_vect_ratio = 0.5*sum(vect_num)/train_num
#
#     tPred = clf.predict(xTest)
#     evalBCResult(tPred, tTest)
#
#     # Visualize data and classification error (and decision boundary)
#     plt.subplot2grid((1, 3), (0, i), rowspan=1, colspan=1)
#
#     plt.scatter(support_vect[:,0],support_vect[:,1],c='k',s=20,label="Support Vectors",facecolors=None)
#
#     disp2DResult(xTrain, np.column_stack((1 - tTrain, tTrain)).astype(bool),0,label_ext='Train')
#     disp2DResult(xTest, np.column_stack((1 - tTest, tTest)).astype(bool),0,label_ext='Test')
#
    # xRange = np.arange(xMin,xMax,0.05)
    # yRange = np.arange(yMin,yMax,0.05)
#     xGrid, yGrid = np.meshgrid(xRange, yRange, sparse=False, indexing='xy')
#     xGrid = np.reshape(xGrid, (xGrid.size,1))
#     yGrid = np.reshape(yGrid, (yGrid.size,1))
#     deciBoundX = np.column_stack((xGrid,yGrid))
#     classPr = clf.predict(deciBoundX)
#     plt.scatter(deciBoundX[(classPr == 1),0],deciBoundX[(classPr == 1),1],s=0.01,c='g',label='Class 1 Boundary')
#     if (i ==1):
#         (plt.gca()).legend(loc='upper left',bbox_to_anchor=(0.25, -0.15))
#     plt.title("l={0:.1f}, fraction of support vectors = {1:.1f}%" \
#      .format(l[i],100*support_vect_ratio),fontsize=12)
#
# plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.1)
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# 2) Adaboost with decision stumps as weak learners
disp2DResult(xTrain, np.column_stack((1 - tTrain, tTrain)).astype(bool),0)
clf = Adaboost(200)
clf.fit(xTrain, tTrain, dispBound=True)
tPred = clf.predict(xTrain)
print("Train classification error = {:.1f}%".format(100*np.sum(tPred != tTrain)/tTrain.shape[0]))
# tPred = clf.predict(xTest)
# print("Final classification error = {:.1f}%".format(100*np.sum(tPred != tTest)/tTest.shape[0]))

xRange = np.arange(xMin,xMax,0.05)
yRange = np.arange(yMin,yMax,0.05)
xGrid, yGrid = np.meshgrid(xRange, yRange, sparse=False, indexing='xy')
xGrid = np.reshape(xGrid, (xGrid.size,1))
yGrid = np.reshape(yGrid, (yGrid.size,1))
deciBoundX = np.column_stack((xGrid,yGrid))
classPr = clf.predict(deciBoundX)
plt.scatter(deciBoundX[(classPr == 1),0],deciBoundX[(classPr == 1),1],s=0.01,c='g',label='Class 1 Boundary')
(plt.gca()).legend(loc='upper left',bbox_to_anchor=(0.25, -0.15))

plt.show()

#-----------------------------------------------------------------------------------------------------------------------
print('End')

#-----------------------------------------------------------------------------------------------------------------------
# def fastfit(self, data_in, labels):
#     [dataLen, self.d] = np.shape(data_in)
#     class0_ind = (labels==0)
#     class1_ind = (labels==1)
#     class0_mean = (self.weight[class0_ind]).dot(data_in[class0_ind,:])
#     class1_mean = (self.weight[class1_ind]).dot(data_in[class1_ind,:])
#     class0_var = (self.weight[class0_ind]).dot(np.square(data_in[class0_ind,:]-class0_mean))
#     class1_var = (self.weight[class1_ind]).dot(np.square(data_in[class1_ind,:]-class1_mean))
#     var_sum = class0_var + class1_var
#     pred_sign_flip = (class1_mean < class0_mean)
#     decision_bound = np.divide(np.multiply(class0_var, class1_mean), var_sum) + \
#                      np.divide(np.multiply(class1_var, class0_mean), var_sum)
#     pred_label = np.zeros((dataLen,self.d))
#     class1_pred_ind = np.logical_xor((data_in > decision_bound),pred_sign_flip)
#     pred_label[class1_pred_ind] = 1
#
#     fit_err_ind = np.not_equal(pred_label.T,labels)
#     fit_err = np.sum(fit_err_ind,axis=1)/dataLen
#
#     self.slct_dimen= np.argmin(fit_err)
#     fit_err_ind = fit_err_ind[self.slct_dimen, :]
#     if (self.d == 2):
#         self.data_min = np.min(data_in,axis=0)
#         self.data_max = np.max(data_in,axis=0)
#     self.fit_err = fit_err[self.slct_dimen]
#     self.decision_bound = decision_bound[self.slct_dimen]
#     self.pred_sign_flip = pred_sign_flip[self.slct_dimen]
#     return fit_err_ind