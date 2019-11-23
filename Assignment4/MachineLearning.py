import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from colorama import Fore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
import time
import mlrose

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
from sklearn import metrics
import scipy.sparse as sps
from scipy.linalg import pinv
from scipy.stats import kurtosis
import tensorflow as tf
import random
import mdptoolbox, mdptoolbox.example

# todo: learn more about sklearn Validation Curve
# todo: learn more about sklearn GridSearchCV
# todo: learn more about sklearn pipeline
# todo: learn more about sklearn make scorer
# todo: learn more about sklean learning_curve
# todo: learn more about sklearn compute_sample_weight
# todo: learn more about sklean SelectFromModel

class ReinforcementLearning(unittest.TestCase):

    def testGridWorldValue(self):
        P, R = self.gridWorld()
        # pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.98, epsilon=0.01)
        # pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.95, epsilon=0.01)
        pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.90, epsilon=0.01)
        pi.run()
        t = pi.policy

    def testsGridWorldPolicy(self):
        P, R = self.gridWorld()
        # pi = mdptoolbox.mdp.PolicyIteration(P, R, discount = .98)
        pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=0.95)
        pi.run()
        t = pi.policy

    def testForestValue(self):
        P, R = mdptoolbox.example.forest(S=49, r1=4, r2=2, p=0.1)
        pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.95, epsilon=0.01, forest = True)
        # pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.98, epsilon=0.01, forest=True)
        # pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.999, epsilon=0.01, forest=True)
        # pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.9, epsilon=0.01, forest=True)
        # pi = mdptoolbox.mdp.ValueIteration(P, R, discount=.8, epsilon=0.01, forest=True)
        pi.run()
        t = pi.policy

    def testForestPolicy(self):
        P, R = mdptoolbox.example.forest(S=49, r1=4, r2=2, p=0.1)
        pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=.95)
        # pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=.98)
        # pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=.999)
        # pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=.9)
        # pi = mdptoolbox.mdp.PolicyIteration(P, R, discount=.8)
        pi.run()
        t = pi.policy

    def testForestQ(self):
        P, R = mdptoolbox.example.forest(S=49, r1=4, r2=2, p=0.1)
        explore = [1, 2, 3, 5, 10, 20, 50, 200]
        for e in explore:
            pi = mdptoolbox.mdp.QLearning(P, R, discount=.98, exploration=e, n_iter = 1000000, problem = 'notgrid')
            pi.run()
        t = pi.policy

    def testGridWorldQ(self):
        explore = [1, 2, 3, 5, 10, 20, 50, 200]
        P, R = self.gridWorld()
        for e in explore:
            pi = mdptoolbox.mdp.QLearning(P, R, discount = 0.98, exploration=e, n_iter = 3000000)
            pi.run()
        t = pi.policy

    def gridWorld(self):

        width = 50
        S = int(width * width)
        A = 4
        P = np.zeros((A,S,S))
        R = np.zeros((S))
        R[:]=-.005
        R[S-1] = 3
        for i in range(width):
            if i != 24 and i != 25:
                R[49 + 49*i]= -10
        for i in range(7):
            R[44+49*(i+19)]= -10

        for a in range(A):
            for i in range(width): # columns, x
                for j in range(width): # rows, y
                    up = j - 1 if j - 1 >= 0 else j+1
                    left = i - 1 if i - 1 >= 0 else i+1
                    right = i + 1 if i + 1 < width else i-1
                    down = j + 1 if j + 1 < width else j-1
                    if a == 0:
                        P[a, width * j + i, width * up + i] += .8
                        P[a, width * j + i, width * j + left] += .1
                        P[a, width * j + i, width * j + right] += .1
                    elif a==1:
                        P[a, width * j + i, width * j + left] += .8
                        P[a, width * j + i, width * up + i] += .1
                        P[a, width * j + i, width * down + i] += .1
                    elif a == 2:
                        P[a, width * j + i, width * j + right] += .8
                        P[a, width * j + i, width * up + i] += .1
                        P[a, width * j + i, width * down + i] += .1
                    elif a == 3:
                        P[a, width * j + i, width * down + i] += .8
                        P[a, width * j + i, width * j + left] += .1
                        P[a, width * j + i, width * j + right] += .1
        return (P, R)

    def largeGridWorld(self):
        width = 50
        S = int(width * width)
        A = 4
        P = np.zeros((A,S,S))
        R = np.zeros((S))
        R[:]=-.005
        R[S-1] = 2
        for i in range(50):
            # if i != 23 and i != 24 and i != 25 and i != 26 and i != 27:
            R[49 + 49 * i] = -10

        for a in range(A):
            for i in range(width): # columns
                for j in range(width): # rows
                    up = j-1 if j-1 >=0 else j+1
                    left = i-1 if i-1 >=0 else i+1
                    right = i+1 if i+1<width else i-1
                    down = j+1 if j+1<width else j-1
                    if a == 0:
                        P[a,width*j+i, width*up+i] += .8
                        P[a, width * j + i, width * j + left] += .1
                        P[a, width * j + i, width * j + right] += .1
                    elif a==1:
                        P[a, width * j + i, width * j + left] += .8
                        P[a, width * j + i, width * up + i] += .1
                        P[a, width * j + i, width * down + i] += .1
                    elif a == 2:
                        P[a, width * j + i, width * j + right] += .8
                        P[a, width * j + i, width * up + i] += .1
                        P[a, width * j + i, width * down + i] += .1
                    elif a == 3:
                        P[a, width * j + i, width * down + i] += .8
                        P[a, width * j + i, width * j + left] += .1
                        P[a, width * j + i, width * j + right] += .1
        return (P, R)