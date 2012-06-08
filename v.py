import datetime, os, time
import numpy as np
import pylab as pl

#from calc_tsne import calc_tsne as tsne
from tsne import tsne 

X = np.array( [pl.imread('frames/' + f).flatten() for f in sorted(os.listdir("frames"))], dtype = np.float64 )

SW = np.random.permutation(X.shape[1])
X = X[:, SW]     # permute pixels

results = tsne(X.T)
x,y = results.real[:,0], results.real[:,1]

pl.scatter(x, y, c = np.arange(X.shape[1])[SW], cmap=pl.jet())
pl.show()


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
