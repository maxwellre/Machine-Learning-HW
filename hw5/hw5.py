# Homework 3: Unsupervised Learning, part 1 (Gaussian mixtures and EM algorithm; K-means and soft K-means
# Created on 05/06/2018
# Author: Yitian Shao
#-----------------------------------------------------------------------------------------------------------------------
import contextlib
import numpy as np
import numpy.matlib as matlib
import numpy.linalg as nla
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#-----------------------------------------------------------------------------------------------------------------------
# Configurations
np.random.seed(0)

#-----------------------------------------------------------------------------------------------------------------------
# Functions
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

def draw_ellipse(position, covariance, ax=None, **kwargs):
    ax = ax or plt.gca()
    if covariance.shape == (2, 2):
        U, s, Vt = nla.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))

def gaussianFromEigen(mean_v, lamb, eig_vectors, data_num):
    dimen = eig_vectors.shape[0]
    Cov = matlib.zeros((dimen, dimen))
    for i in range(dimen):
        Cov = Cov +( lamb[i]* eig_vectors[:,i]* (eig_vectors[:,i].T) )
    ret_data = np.random.multivariate_normal(mean_v, Cov, data_num)
    return ret_data, Cov

def getEmpProbTable(comp_num, k, true_label, pred_label, margin_prob):
    joint_prob = np.zeros((comp_num,k)) # P(a,z)
    for c_i in range(k):
        ind = (pred_label == c_i)
        if c_i == 0:
            curr_clust = ind
        else:
            curr_clust = np.column_stack((curr_clust, ind))

        for a_i in range(comp_num):
            joint_prob[a_i,c_i] = np.mean(np.logical_and(true_label[:,a_i],ind))
    emp_prob = joint_prob/margin_prob # # empirical probabilities P(a|z) = P(a,z)/P(z)
    with printoptions(precision=3, suppress=True):
        print("Empirical probability table:\n",emp_prob)
    return curr_clust

# Visualize data
def disp2DResult(data_in, one_hot, disp_now=1):
    label_num = one_hot.shape[1]
    for i in range(label_num):
        plt.scatter(data_in[one_hot[:,i],0],data_in[one_hot[:,i],1],s=5,label="Component {:2d}".format(i+1))
    plt.title('Data visulization',fontsize=12)
    plt.xlabel('Dimension 0',fontsize=10)
    plt.ylabel('Dimension 1',fontsize=10)
    plt_ax = plt.gca()
    plt_ax.set_aspect('equal', 'box')
    plt_ax.legend(loc='upper left',bbox_to_anchor=(-0.3, 1.2))
    if disp_now:
        plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Generate seven "quasi-orthogonal" random vectors in d dimensions
d = 30 # increased dimension
Pu0 = 2/3 # P(u[i] = 0)
Pu1 = 1/6 # P(u[i] = +1)
Pu2 = 1/6 # P(u[i] = -1)
vect_pool_size = 7
not_ortho = 1
vect_num = 0

vect_pool = np.random.choice([0,1,-1], size=[7,d], p=[Pu0, Pu1, Pu2])

ortho_thresh = 1 # threshold can be adjusted
diag_ind = np.diag_indices(vect_pool_size)
iter_i = 0
while not_ortho:
    dot_prod = (vect_pool).dot(vect_pool.T)
    dot_prod[diag_ind] = 0
    corr_score = np.sum(np.absolute(dot_prod),axis=1)
    # print("iter ",iter_i,"corr_score = ",corr_score)
    if (np.max(corr_score) == 0) or (np.sum(corr_score) < ortho_thresh):
        not_ortho = 0
    else:
        iter_i += 1
        drop_ind = np.argmax(corr_score)
        vect_pool[drop_ind,:] = np.random.choice([0,1,-1], d, p=[Pu0, Pu1, Pu2])
print("[u1,...,u7] = \n",vect_pool.T)

#-----------------------------------------------------------------------------------------------------------------------
# Generate 100-dimensional data samples for a Gaussian mixture distribution with 3 equiprobable components

sigma = 0.01 # noise level
data_num = 200 # data number to generate

Z1 = np.random.normal(0, 1, data_num)
Z2 = np.random.normal(0, 1, data_num)
N = np.random.normal(0, sigma, size=[data_num,d])

Comp1 = vect_pool[0,:] + np.outer(Z1,vect_pool[1,:]) + np.outer(Z2,vect_pool[2,:]) + N
Comp2 = 2*vect_pool[3,:] + (2**0.5)*np.outer(Z1,vect_pool[4,:]) + np.outer(Z2,vect_pool[5,:]) + N
Comp3 = (2**0.5)*vect_pool[5,:] + np.outer(Z1,(vect_pool[0,:]+vect_pool[1,:])) + \
        0.5*(2**0.5)*np.outer(Z2,vect_pool[4,:]) + N

# (E[X])
Comp_mean = np.zeros((3,d))
Comp_mean[0,:] = vect_pool[0,:]
Comp_mean[1,:] = 2*vect_pool[3,:]
Comp_mean[2,:] = (2**0.5)*vect_pool[5,:]

# (Var(X) = E[X^2] - E[X]^2 => Var(X1+X2) = Var(X1) + Var(X2) if X1 and X2 are independent)
Comp_var = np.zeros((3,d))
Comp_var[0,:] = np.square(vect_pool[1,:]) + np.square(vect_pool[2,:]) + (sigma**2)*np.ones(d)
Comp_var[1,:] = 2*np.square(vect_pool[4,:]) + np.square(vect_pool[5,:]) + (sigma**2)*np.ones(d)
Comp_var[2,:] = np.square(vect_pool[0,:]+vect_pool[1,:]) + 0.5*np.square(vect_pool[4,:]) + (sigma**2)*np.ones(d)
# Comp_eig_vect = np.sqrt(Comp_var)
# print("Component covariance matrix eigenvector (transposed): \n", Comp_eig_vect.T)

mixGaussInd = np.random.choice(3,data_num)
ind0 = (mixGaussInd==0)
ind1 = (mixGaussInd==1)
ind2 = (mixGaussInd==2)
x0Len = np.sum(ind0)
x1Len = np.sum(ind1)
x2Len = np.sum(ind2)
prob_z = np.array([x0Len, x1Len, x2Len]).reshape((3,1))/data_num # P(z)
print("Ratio of gaussian mixture components (30D) = {:d}:{:d}:{:d}".format(x0Len,x1Len,x2Len))
print("P(z[0]=1) = {:.3f}, P(z[1]=1) = {:.3f}, P(z[2]=1) = {:.3f}".format(prob_z[0,0],prob_z[1,0],prob_z[2,0]))
x30D = np.concatenate((Comp1[ind0,:], Comp2[ind1,:], Comp3[ind2,:]),axis=0)
temp = np.concatenate((np.zeros(x0Len),np.ones(x1Len),2*np.ones(x2Len)))
z30D = np.column_stack((temp==0,temp==1,temp==2)) # One-hot encoding (.astype(int))

#-----------------------------------------------------------------------------------------------------------------------
pass