from sklearn.cluster import MiniBatchKMeans
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

def extMask(cube, threshold):

    '''
    Extract salt/horizon mask from an attribute cube
    '''
    # #apply PCA to reduce dimensionality
    # from sklearn.decomposition import KernelPCA
    # cube = KernelPCA(n_components=2, kernel='rbf').fit_transform(cube.squeeze())
    geobody = np.where(cube < threshold, 255, 0).astype('int32') #depends on slice type

    return geobody

def kMeans(attri, nclusters=2):
    '''
    To cluster attribute array into horizon/non-horizon  
    '''

    kmeans = MiniBatchKMeans(n_clusters=nclusters, random_state=0).fit_predict(attri.reshape((-1, 1)))

    k_pred = kmeans.reshape(*attri.shape)
    
    return k_pred