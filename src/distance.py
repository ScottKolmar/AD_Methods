import numpy as np
from scipy import spatial

####################
# DISTANCE FUNCTIONS
####################

class Distance:

    def euclidean(v1, v2):
        distance = np.sqrt(np.sum((v1-v2)**2))
        return distance

    def cityblock(v1, v2):
        distance = np.sum(abs(v1-v2))
        return distance

    def minkowski(v1, v2, n=2):
        distance = np.sum(abs(v1-v2)**n)**(1/n)
        return distance

    def chebyshev(v1, v2):
        distance = np.max(abs(v1-v2))
        return distance

    def sorensen(v1, v2):
        A = np.sum(abs(v1-v2))
        d = len(v1)
        distance = A/np.sum(v1+v2)
        return distance

    def gower(v1, v2):
        A = np.sum(abs(v1-v2))
        d = len(v1)
        distance = A/d
        return distance

    def sorgel(v1, v2):
        A = np.sum(abs(v1-v2))
        d = len(v1)
        distance = A/np.sum(np.maximum(v1,v2))
        return distance

    def kulczynski(v1, v2):
        A = np.sum(abs(v1-v2))
        d = len(v1)
        distance = A/np.sum(np.minimum(v1,v2))
        return distance

    def canberra(v1,v2):
        distance = spatial.distance.canberra(v1,v2)
        return distance

    def lorentzian(v1,v2):
        A = np.sum(abs(v1-v2))
        d = len(v1)
        distance = np.sum(np.log(1+abs(v1-v2)))
        return distance

    def czekanowski(v1,v2):
        A = np.sum(abs(v1-v2))
        distance = A/np.sum(v1+v2)
        return distance

    def ruzicka(v1,v2):
        distance = 1-np.sum(np.minimum(v1,v2))/np.sum(np.maximum(v1,v2))
        return distance

    def tanimoto(v1,v2):
        distance = np.sum(np.maximum(v1,v2)-np.minimum(v1,v2))/np.sum(np.maximum(v1,v2))
        return distance

    def cosine_distance(v1,v2):
        distance = spatial.distance.cosine(v1,v2)
        return distance