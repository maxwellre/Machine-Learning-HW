# Homework 5: Sparsity (PCA and Compressive Sensing)
# Created on 06/06/2018
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso

#-----------------------------------------------------------------------------------------------------------------------
# Configurations
np.random.seed(11)

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
    with printoptions(precision=1, suppress=True):
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

# Normalized mean square error (MSE)
def MSE(y, recon):
    return (nla.norm(y - recon)/nla.norm(y))**2

#-----------------------------------------------------------------------------------------------------------------------
# Generate seven "quasi-orthogonal" random vectors in d dimensions
d = 100 # increased dimension
Pu0 = 2/3 # P(u[i] = 0)
Pu1 = 1/6 # P(u[i] = +1)
Pu2 = 1/6 # P(u[i] = -1)
vect_pool_size = 6
not_ortho = 1
vect_num = 0

vect_pool = np.random.choice([0,1,-1], size=[vect_pool_size,d], p=[Pu0, Pu1, Pu2])

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
print("[u1,...,u6] = \n",vect_pool.T)

#-----------------------------------------------------------------------------------------------------------------------
# 1.Generate 100-dimensional data samples for a Gaussian mixture distribution with 3 equiprobable components
sigma = 0.01 # noise level
data_num = 200 # data number to generate

Z1 = np.random.normal(0, 1, data_num)
Z2 = np.random.normal(0, 1, data_num)
N = np.random.normal(0, sigma, size=[data_num,d])

Comp1 = vect_pool[0,:] + np.outer(Z1,vect_pool[1,:]) + np.outer(Z2,vect_pool[2,:]) + N
Comp2 = 2*vect_pool[3,:] + (2**0.5)*np.outer(Z1,vect_pool[4,:]) + np.outer(Z2,vect_pool[5,:]) + N
Comp3 = (2**0.5)*vect_pool[5,:] + np.outer(Z1,(vect_pool[0,:]+vect_pool[1,:])) + \
        0.5*(2**0.5)*np.outer(Z2,vect_pool[4,:]) + N

Comp_mean = np.zeros((3,d))
Comp_mean[0,:] = vect_pool[0,:]
Comp_mean[1,:] = 2*vect_pool[3,:]
Comp_mean[2,:] = (2**0.5)*vect_pool[5,:]

mixGaussInd = np.random.choice(3,data_num)
ind0 = (mixGaussInd==0)
ind1 = (mixGaussInd==1)
ind2 = (mixGaussInd==2)
x0Len = np.sum(ind0)
x1Len = np.sum(ind1)
x2Len = np.sum(ind2)
prob_z = np.array([x0Len, x1Len, x2Len]).reshape((3,1))/data_num # P(z)
print("Ratio of gaussian mixture components = {:d}:{:d}:{:d}".format(x0Len,x1Len,x2Len))
print("P(z[0]=1) = {:.3f}, P(z[1]=1) = {:.3f}, P(z[2]=1) = {:.3f}".format(prob_z[0,0],prob_z[1,0],prob_z[2,0]))
x = np.concatenate((Comp1[ind0,:], Comp2[ind1,:], Comp3[ind2,:]),axis=0)
temp = np.concatenate((np.zeros(x0Len),np.ones(x1Len),2*np.ones(x2Len)))
z = np.column_stack((temp==0,temp==1,temp==2)) # One-hot encoding (.astype(int))

#-----------------------------------------------------------------------------------------------------------------------
# Part I
# 1a) SVD
u, s, vh = nla.svd(x)
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
plt.plot(range(1,d+1),s)
plt.xlabel("Singular value index")
plt.ylabel("Singular values")
#-----------------------------------------------------------------------------------------------------------------------
# 1b) PCA
domin_num = 6

scaler = StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
mean_scaled = scaler.transform(Comp_mean)

pca = PCA(n_components=domin_num)
pca.fit(x_scaled)
x_projected = pca.transform(x_scaled)
mean_proj = pca.transform(mean_scaled)
with printoptions(precision=1, suppress=True):
    print("Projections of the component mean:\n",mean_proj)

plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x860+20+20")
plt.subplot(1,2,1)
disp2DResult(x_scaled[:,:2], z, disp_now=0)
plt.plot(mean_scaled[:, 0], mean_scaled[:, 1], 'm^')
plt.subplot(1,2,2)
disp2DResult(x_projected[:,:2], z, disp_now=0)
plt.plot(mean_proj[:, 0], mean_proj[:, 1], 'm^')

print("------------------------- K-means of data -------------------------")
clust_center = []
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.05, hspace=0.3)
for k in range(2,6):
    print("k = ",k)
    kmean_h = KMeans(n_clusters=k, init='random', n_init=10, random_state=0).fit(x_projected)
    clust_center.append(kmean_h.cluster_centers_)

    with printoptions(precision=1, suppress=True):
        print("Cluster center:\n", kmean_h.cluster_centers_)

    curr_clust = getEmpProbTable(3, k, z, kmean_h.labels_, prob_z)

    plt.subplot(gs[k - 2])
    disp2DResult(x_projected[:, :2], curr_clust, 0)
    plt.title("{:d} Clusters".format(k))
    plt.plot(clust_center[k - 2][:, 0], clust_center[k - 2][:, 1], 'kx',label="KMean Center")
    plt.plot(mean_proj[:, 0], mean_proj[:, 1], 'm^')
    plt.legend(loc='upper left', bbox_to_anchor=(-1.0, 1.2))

#-----------------------------------------------------------------------------------------------------------------------
# 2) How the cluster centers found by K-means relate to the d0-dimensional projections of the vectors
for i in range(4):
    clust_vect_corr = (mean_proj/nla.norm(mean_proj,axis=1).reshape((mean_proj.shape[0],1))) \
        .dot((clust_center[i]/nla.norm(clust_center[i],axis=1).reshape((clust_center[i].shape[0],1))).T)
    with printoptions(precision=1, suppress=True):
        print("k={:2d}: Correlation between cluster mean and data model mean:\n".format(i+2),clust_vect_corr)

#-----------------------------------------------------------------------------------------------------------------------
# Part II
# 3) Generate a m-by-d matrix Phi
m = 4 # Projection dimension
P0 = 1/2 # P(u[i] = +1)
P1 = 1/2 # P(u[i] = -1)

m_factor = 1/np.sqrt(m)
Phi = np.random.choice([1,-1], size=[m,d], p=[P0, P1])
#-----------------------------------------------------------------------------------------------------------------------
# 3a) Compressive projection:
y =  m_factor * np.matmul(Phi,x.T) # Size of y: [m,data_num]

# 3b) Basis
B = vect_pool

#-----------------------------------------------------------------------------------------------------------------------
# 4) Sparse reconstruction of the data
PhiB = m_factor * np.matmul(Phi,B.T)
clf = Lasso(alpha = 1.0)
clf.fit(PhiB,y)
a_hat = clf.coef_
with printoptions(precision=1, suppress=True):
    print(a_hat)

recon = np.matmul(PhiB,a_hat.T)
Lasso_loss = MSE(y, recon)
# with printoptions(precision=1, suppress=True):
#     print("y = \n",y, "\ny Reconstruction = \n",recon)
print("Lasso reconstruction normalized MSE = {:.2f}".format(Lasso_loss))
#-----------------------------------------------------------------------------------------------------------------------
# 5) normalized MSE versus averaged over all the data.
draw_num = 10
lamb = np.arange(1,41,dtype='float')/20.0
lamb_num = len(lamb)
norm_MSE = np.empty((draw_num,lamb_num))
MSE_Comp1 = np.empty((draw_num,lamb_num))
MSE_Comp2 = np.empty((draw_num,lamb_num))
MSE_Comp3 = np.empty((draw_num,lamb_num))
for i in range(draw_num):
    Z1 = np.random.normal(0, 1, data_num)
    Z2 = np.random.normal(0, 1, data_num)
    N = np.random.normal(0, sigma, size=[data_num,d])
    Comp1 = vect_pool[0,:] + np.outer(Z1,vect_pool[1,:]) + np.outer(Z2,vect_pool[2,:]) + N
    Comp2 = 2*vect_pool[3,:] + (2**0.5)*np.outer(Z1,vect_pool[4,:]) + np.outer(Z2,vect_pool[5,:]) + N
    Comp3 = (2**0.5)*vect_pool[5,:] + np.outer(Z1,(vect_pool[0,:]+vect_pool[1,:])) + \
            0.5*(2**0.5)*np.outer(Z2,vect_pool[4,:]) + N
    mixGaussInd = np.random.choice(3,data_num)
    ind0 = (mixGaussInd==0)
    ind1 = (mixGaussInd==1)
    ind2 = (mixGaussInd==2)
    x0Len = np.sum(ind0)
    x1Len = np.sum(ind1)
    x2Len = np.sum(ind2)
    prob_z_temp = np.array([x0Len, x1Len, x2Len]).reshape((3,1))/data_num # P(z)
    print("Draw {:d}\nRatio of gaussian mixture components = {:d}:{:d}:{:d}".format(i,x0Len,x1Len,x2Len))
    print("P(z[0]=1) = {:.3f}, P(z[1]=1) = {:.3f}, P(z[2]=1) = {:.3f}"
          .format(prob_z_temp[0,0],prob_z_temp[1,0],prob_z_temp[2,0]))
    x_temp = np.concatenate((Comp1[ind0,:], Comp2[ind1,:], Comp3[ind2,:]),axis=0)

    y_temp = m_factor * np.matmul(Phi, x_temp.T)

    for j in range(lamb_num):
        clf = Lasso(alpha=lamb[j])
        clf.fit(PhiB, y_temp)
        a_hat = clf.coef_
        recon = np.matmul(PhiB, a_hat.T)
        norm_MSE[i, j] = MSE(y_temp, recon)
        MSE_Comp1[i, j] = MSE(y_temp[:,ind0], recon[:,ind0])
        MSE_Comp2[i, j] = MSE(y_temp[:, ind1], recon[:, ind1])
        MSE_Comp3[i, j] = MSE(y_temp[:, ind2], recon[:, ind2])

plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x660+20+20")
plt.subplot(1,2,1)
plt.plot(lamb,np.mean(norm_MSE,axis=0))
plt.xlabel("Lambda")
plt.ylabel("Avg. Normalized MSE")
plt.subplot(1,2,2)
plt.plot(lamb,np.mean(MSE_Comp1,axis=0),label="Component {:2d}".format(1))
plt.plot(lamb,np.mean(MSE_Comp2,axis=0),label="Component {:2d}".format(2))
plt.plot(lamb,np.mean(MSE_Comp3,axis=0),label="Component {:2d}".format(3))
plt_ax = plt.gca()
plt_ax.legend(loc='upper left')
plt.xlabel("Lambda")
plt.ylabel("Avg. Normalized MSE")

#-----------------------------------------------------------------------------------------------------------------------
# 6)
Euclid = np.zeros((vect_pool_size,vect_pool_size))
Euclid_proj = np.zeros((vect_pool_size,vect_pool_size))
for i in range(vect_pool_size-1):
    for j in range(i+1,vect_pool_size):
        Euclid[i,j] = nla.norm(B[i,:]-B[j,:])
        Euclid_proj[i, j] = nla.norm(PhiB[:,i]-PhiB[:,j])
with printoptions(precision=1, suppress=True):
    print("Euclidean distance between u:\n",Euclid,"\nEuclidean distance between projected u:\n",Euclid_proj)

#-----------------------------------------------------------------------------------------------------------------------
# 7,8) K-means algorithm post-projection
# How the cluster centers found by K-means relate to the m-dimensional projections of the vectors
mean_m_proj =  m_factor * np.matmul(Phi,Comp_mean.T).T

plt.figure()
disp2DResult(y[:2,:].T, z, disp_now=0)
plt.plot(mean_m_proj[:, 0], mean_m_proj[:, 1], 'm^')

print("------------------------- K-means of the projected data -------------------------")
clust_center = []
plt.figure()
plt.get_current_fig_manager().window.wm_geometry("1400x760+20+20")
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.05, hspace=0.3)
for k in range(2,6):
    print("k = ",k)
    kmean_h = KMeans(n_clusters=k, init='random', n_init=10, random_state=0).fit(y.T)
    clust_center.append(kmean_h.cluster_centers_)

    with printoptions(precision=1, suppress=True):
        print("Cluster center:\n", kmean_h.cluster_centers_)

    curr_clust = getEmpProbTable(3, k, z, kmean_h.labels_, prob_z)

    plt.subplot(gs[k - 2])
    disp2DResult(y[:2,:].T, curr_clust, 0)
    plt.title("{:d} Clusters".format(k))
    plt.plot(clust_center[k - 2][:, 0], clust_center[k - 2][:, 1], 'kx',label="KMean Center")
    plt.plot(mean_m_proj[:, 0], mean_m_proj[:, 1], 'm^')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.7, 1.2))

for i in range(4):
    clust_vect_corr = (mean_m_proj/nla.norm(mean_m_proj,axis=1).reshape((mean_m_proj.shape[0],1))) \
        .dot((clust_center[i]/nla.norm(clust_center[i],axis=1).reshape((clust_center[i].shape[0],1))).T)
    with printoptions(precision=1, suppress=True):
        print("k={:2d}: Correlation between cluster mean and data model mean:\n".format(i+2),clust_vect_corr)

#-----------------------------------------------------------------------------------------------------------------------
plt.show()