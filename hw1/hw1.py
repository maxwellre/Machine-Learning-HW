# Homework 1: Classification using logistic regression
# Created on 04/08/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import scipy.stats as scist

# Configurations
dimen = 2

#-----------------------------------------------------------------------------------------------------------------------
# Functions
def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    Cov = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        Cov = Cov +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, Cov, data_num)
    return ret_data, Cov

# Implementation of scist.multivariate_normal.pdf(xi, mean_v, Cov)
def getGaussianLikelihood(x_i, mean_v, Cov):
    dimen = Cov.shape[0]
    GaussZ = np.power(2 * np.pi, (dimen * 0.5)) * np.power(np.linalg.det(Cov), 0.5)
    likelihood = np.exp(-0.5 * np.mat(x_i - mean_v) * (np.linalg.inv(Cov)) * (np.mat(x_i - mean_v).T)) / GaussZ
    return likelihood

# Implement the MAP decision rule
def binaryMAP(data_in):
    classPr = np.empty(shape=[0, dimen])
    for xi in data_in:
        Pr_t0 = getGaussianLikelihood(xi, m0, C0)

        Pr_t1A = getGaussianLikelihood(xi, mA, C1A)
        Pr_t1B = getGaussianLikelihood(xi, mB, C1B)
        Pr_t1 = pi_A * Pr_t1A + pi_B * Pr_t1B

        classPr = np.append(classPr,np.concatenate((Pr_t0,Pr_t1),axis=1),axis=0)

    return classPr

#-----------------------------------------------------------------------------------------------------------------------
# 1) Generating data
data_num = 200

# Class 0
theta0 = 0
m0 = np.zeros(dimen)
lamb0 = [2,1]
U0 = np.mat([[np.cos(theta0), np.sin(theta0)],[-np.sin(theta0), np.cos(theta0)]]).T

x0, C0 = gaussianFromEigen(m0, lamb0, U0, data_num)

# Class 1
thetaA = -3*(np.pi)/4
pi_A = 1/3
mA = [-2,1]
lambA = [2,1/4]
UA = np.mat([[np.cos(thetaA), np.sin(thetaA)],[-np.sin(thetaA), np.cos(thetaA)]]).T

thetaB = (np.pi)/4
pi_B = 2/3
mB = [3,2]
lambB = [3,1]
UB = np.mat([[np.cos(thetaB), np.sin(thetaB)],[-np.sin(thetaB), np.cos(thetaB)]]).T

x1A, C1A = gaussianFromEigen(mA, lambA, UA, data_num)
x1B, C1B = gaussianFromEigen(mB, lambB, UB, data_num)
mixGaussPi = np.random.uniform(0.0,1.0,data_num)

x1 = np.concatenate((x1A[mixGaussPi <= pi_A,:],x1B[mixGaussPi > pi_A,:]),axis=0)

print('C0 =\n',C0, '\n C1A =\n',C1A,'\n C1B =\n',C1B)

x = np.concatenate((x0,x1),axis=0) # data
t = np.concatenate((np.zeros(data_num),np.ones(data_num))) # label
#-----------------------------------------------------------------------------------------------------------------------
# 2)
# xRange = np.arange(np.min(x[:,0]),np.max(x[:,0]),0.05)
# yRange = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.05)
# xGrid, yGrid = np.meshgrid(xRange, yRange, sparse=False, indexing='xy')
# xGrid = np.reshape(xGrid, (xGrid.size,1))
# yGrid = np.reshape(yGrid, (yGrid.size,1))
# deciBoundX = np.column_stack((xGrid,yGrid))
# classPr = binaryMAP(deciBoundX)
# boundToler = 0.005;
# boundInd = ( np.abs(classPr[:,1]-classPr[:,0]) < boundToler)

classPr = binaryMAP(x)
tPredit = ( (classPr[:,1]-classPr[:,0]) >= 0 ).astype(int)
incorrInd = np.squeeze(np.asarray((tPredit.flatten() != t)))

# Visualize data and decision boundary
plt.figure(figsize=(12, 9))
# plt.scatter(xGrid[boundInd],yGrid[boundInd],s=0.5,c='g')
plt.scatter(x0[:,0],x0[:,1],s=5,label='Class 0')
plt.scatter(x1[:,0],x1[:,1],s=5,c='r',label='Class 1')
plt.scatter(x[incorrInd,0],x[incorrInd,1],s=16,facecolors='none',edgecolors='k',label='Incorrect Classification')
plt.title('MAP Classifier',fontsize=12)
plt.xlabel('Dimension 0',fontsize=10)
plt.ylabel('Dimension 1',fontsize=10)
ax1 = plt.gca()
ax1.set_aspect('equal', 'box')
ax1.legend(loc='upper left')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
print('End')