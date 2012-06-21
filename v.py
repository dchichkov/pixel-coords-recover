import datetime, os, time
import numpy as np
import pylab as pl

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances 
from sklearn.decomposition import KernelPCA
from sklearn.utils.graph import graph_shortest_path



def kneighbors(X, n_neighbors, metric):
     """Finds the K-neighbors of a point.
     Based on sklearn
     Returns distance ind
    """
     if metric == 'abs_correlation': 
         dist = pairwise_distances(X, metric = 'correlation')
         dist = np.sqrt( -np.log( np.abs(1 - dist )) )
     else:     
         dist = pairwise_distances(X, metric = metric)
         
         
     neigh_ind = dist.argsort(axis=1)
     neigh_ind = neigh_ind[:, :n_neighbors]
     return neigh_ind

#     
def kneighbors_graph(X, n_neighbors, metric):
     """Computes the (weighted) graph of k-Neighbors for points in X.
     Based on sklearn
     """
     X = np.asarray(X)

     n_samples1 = n_samples2 = X.shape[0]
     n_nonzero = n_samples1 * n_neighbors
     A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

     # construct CSR matrix representation of the k-NN graph
     A_data = np.ones((n_samples1, n_neighbors))
     A_ind = kneighbors(X, n_neighbors, metric)

     return csr_matrix((A_data.ravel(), A_ind.ravel(), A_indptr), shape=(n_samples1, n_samples2))





def isomap(X, n_neighbors, metric):
    """
        Based on sklearn,
        Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
        License: BSD, (C) 2011
    """    
    
    kng = kneighbors_graph(D, n_neighbors = n_neighbors, metric = metric)    
    dist_matrix_ = graph_shortest_path(kng, method='auto', directed=False)    
    kernel_pca_ = KernelPCA(n_components=2, kernel="precomputed", eigen_solver='auto')
    G = dist_matrix_ ** 2
    G *= -0.5
    return kernel_pca_.fit_transform(G)



X = np.array( [pl.imread('frames/' + f).flatten() for f in sorted(os.listdir("frames"))], dtype = np.float64 )

SW = np.random.permutation(X.shape[1])
X = X[:, SW]     # permute pixels


#D = np.array([np.concatenate((pixel, pixel[i%2:-i%2 or None], pixel[(i+1)%2:(i+1)%2 or None])) for i,pixel in enumerate(X.T)])
D = X.T

for metric in ['abs_correlation', 'braycurtis', 'canberra', 'correlation', 'cosine', 'minkowski', 'seuclidean']:
  for n_neighbors in [4,5,6]:
    pl.figure()
    results = isomap(D, n_neighbors, metric = metric)
    x,y = results.real[:,0], results.real[:,1]  
    pl.scatter(x, y, c = np.arange(X.shape[1])[SW], cmap=pl.jet())
    pl.title("%s n_neighbors %d" % (metric, n_neighbors))
    #pl.savefig("%s.n_neighbors.%d.png" % (metric, n_neighbors))
    pl.show()
    

exit()
print "Saving restored frames..."

name = "movie"    
if not os.path.exists(name): os.mkdir(name)
for i in range(len(X)):
    pl.figure(figsize=(2, 2))
    pl.hexbin(x, y, gridsize = 50, C = X[i], cmap=pl.gray())
    pl.savefig("%s/frame.%04d.png" % (name, i))
    pl.close()    
    print ".",
print "Done." 
    
print "Creating movie..."
os.system("ffmpeg -y -f image2 -i %s/frame.%%04d.png -r 12 %s.mov" % (name, name))
