#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import warnings
from random import randint
np.random.seed(42)

class StatScore:
    
    #initialize the class
    def __init__(self,n_boost = 100000, mode='max',verbosity=False):
        self.mode = mode
        
        if not((mode == 'max') | (mode == 'min')):
            raise ValueError('Parameter "Mode" should be equal to "max" or "min"')
            
        
        self.n_boost = int(n_boost)
        if verbosity:
            print("The number of sub-samples : ",n_boost)
            print("Mode: ",mode)
    
    #evaluate the score
    def evaluate(self,scoreVector):
        
        #check the shape of score array
        if ((scoreVector.shape[0] == 1) | (scoreVector.ndim != 1)):
            raise ValueError("Error: the shape of the validation score should be a 1-dimensional vector with shape (n,)")
            return Nan,Nan
        else:
            
            #convert if required, according to the optimization metric
            if (self.mode == 'min'):
                scoreVector = 1/scoreVector;



            scoreVecShape = scoreVector.shape[0]

            #sub-sample array
            vectorSubSample = np.zeros(shape=(self.n_boost,scoreVecShape))

            #apply bootstrap
            for r in range(self.n_boost):
                count=0
                for t in range(scoreVecShape):
                    vectorSubSample[r,t] = scoreVector[np.random.randint(0, scoreVecShape)]


            #distribution of the mean
            meanDist = np.zeros(shape=(self.n_boost,1))

            # distribution of the std
            stdDist = np.zeros(shape=(self.n_boost, 1))

            for r in range(self.n_boost):
                meanDist[r] = np.mean(vectorSubSample[r,:])
                stdDist[r] = np.std(vectorSubSample[r, :])


            #get mean of the distribution:
            val_precision = (np.mean(meanDist))

            #evaluate the statistical significance according to 3-sigma rule:
            val_recall = 1/(np.mean(stdDist)+1)


            return val_precision / val_recall,val_precision

