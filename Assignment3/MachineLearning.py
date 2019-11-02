import unittest
import pandas as pd
import numpy as np
import sys
from colorama import Fore
import matplotlib.pyplot as plt
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


# todo: learn more about sklearn Validation Curve
# todo: learn more about sklearn GridSearchCV
# todo: learn more about sklearn pipeline
# todo: learn more about sklearn make scorer
# todo: learn more about sklean learning_curve
# todo: learn more about sklearn compute_sample_weight
# todo: learn more about sklean SelectFromModel

# Notes for ML4T:
# Adaboost takes a long time to run,  Maybe instead consider ensemble.  Currently not using ensemble.
# KNN has performed surprisingly well with low numbers of leafs
# I need to automate searching through hyperparameters
# Use cross validation instead of the 6 chart methods
# The performance measure (score) does not have to be return based on trades.  I can use squared error instead.
# keep in mind though that the output do not match 1 to 1.  Prediction is much lower.
# Consider using accuracy score from sklearn.

def crossValidation(all_x_data, all_y_data, learner):
    # Run the selected model for each type of algorithm
    training_accuracy = []
    test_accuracy = []
    for i in range(1):
        xTrain, xTest, yTrain, yTest = train_test_split(all_x_data, all_y_data, test_size=0.3, random_state=i)
        temp_train, temp_test = evaluateLeaner(xTrain, yTrain, xTest, yTest, learner)
        training_accuracy.append(temp_train)
        test_accuracy.append(temp_test)
    training_accuracy = sum(training_accuracy) / len(training_accuracy)
    test_accuracy = sum(test_accuracy) / len(test_accuracy)
    # sys.stdout.write(Fore.BLACK + "\nTrain Cross Validation Accuracy: " + Fore.BLUE + "{:.2%}".format(training_accuracy))
    # sys.stdout.write(Fore.BLACK + "\nTest Cross Validation Accuracy: " + Fore.BLUE + "{:.2%}".format(test_accuracy) + "\n")
    return training_accuracy, test_accuracy
def evaluateLeaner(xTrain, yTrain, xTest, yTest, learner):
    #### Run the model on the Trainng data

    # y_train_hot = self.mytransform(yTrain)
    # y_test_hot = self.mytransform(yTest)

    # scaler = MinMaxScaler(feature_range=[-1,1])
    # X_train_scaled = scaler.fit_transform(xTrain)
    # X_test_scaled = scaler.transform(xTest)
    # one_hot = OneHotEncoder(categories='auto')
    # y_train_hot = one_hot.fit_transform(yTrain.to_numpy().reshape(-1, 1)).todense()
    # y_test_hot = one_hot.fit_transform(yTest.to_numpy().reshape(-1, 1)).todense()

    learner.fit(xTrain, yTrain)
    # yPredictTraining = pd.DataFrame(data=learner.predict(xTrain), index = yTrain.index)
    y_train_pred = learner.predict(xTrain)
    training_accuracy = accuracy_score(yTrain, y_train_pred)
    # sys.stdout.write(Fore.BLACK + "\nTraining Accuracy: " + Fore.RED + "{:.2%}".format(training_accuracy))

    #### Run the model on the Test data
    # yPredictTest = pd.DataFrame(data=learner.predict(xTest), index=yTest.index)
    y_test_pred = learner.predict(xTest)
    test_accuracy = accuracy_score(yTest, y_test_pred)
    # sys.stdout.write(Fore.BLACK + "\nTest Accuracy: " + Fore.RED + "{:.2%}".format(test_accuracy)+"\n")
    return training_accuracy, test_accuracy
def mytransform(dFrame):
    d = pd.DataFrame(np.zeros((len(dFrame.index), 3)))
    i = 0
    for index, row in dFrame.iterrows():
        if row[0] == -1:
            d.iloc[i, 0] = 1
        if row[0] == 0:
            d.iloc[i, 1] = 1
        if row[0] == 1:
            d.iloc[i, 2] = 1
        i += 1
    return d
def getTechnicalIndicatorData():
    all_data = pd.read_csv('TechnicalIndicators.csv', parse_dates=True)
    all_data = all_data.drop('spy_Price', axis=1).drop('Daily_Return', axis=1)
    all_x_data = all_data.drop('Result', axis=1)
    all_y_data = pd.DataFrame(all_data['Result'], all_data.index)
    return all_x_data, all_y_data, "Technical Indicator Dataset"
def getPokerData():
    all_data = pd.read_csv('poker.csv', parse_dates=True)
    all_x_data = all_data.drop('Class', axis=1)
    transformer = MinMaxScaler(feature_range=(0, 1)).fit(all_x_data)  # fit does nothing.
    all_x_data = transformer.transform(all_x_data)
    all_x_data = pd.DataFrame(all_x_data, index= all_data.index, columns= all_data.iloc[:,:-1].columns)
    all_y_data = pd.DataFrame(all_data['Class'], all_data.index)
    return all_x_data, all_y_data, "Poker Dataset"
def reconstruction_error(projections, x):
    # Borrowed from cmarion
    w = projections.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p@w)@(x.T)).T
    error = np.sum((x.to_numpy()-reconstructed.to_numpy())**2)
    return error


class DimensionalityReduction(unittest.TestCase):
    def testReal(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data -1).iloc[:, 0].to_numpy().transpose()
                clusters = 8
                f_size = (10, 6)
                cm = 'Set3'
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                clusters = 3
                f_size = (30, 18)
                cm = None
            learner = KMeans(n_clusters=clusters, random_state=0)
            t = learner.fit(all_x_data)
            lab = learner.labels_
            # learner.fit(all_x_data)
            # score = learner.inertia_
            # lab = learner.labels_

            pd.plotting.scatter_matrix(all_x_data, c=y, cmap = cm, figsize=f_size, hist_kwds={'bins': 20}, s=60, alpha=.8)
            plt.tight_layout()
            plt.savefig('Real_Scatter_' + name)

            # Scatterplot Example with 2 features selected.
            # t = learner.fit_transform(all_x_data)
            # lab = learner.labels_
            # plt.scatter(all_x_data[:, 5], all_x_data[:, 8], c=lab)
            # plt.legend()
            # plt.show()

    def testKMeans(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()

            resultFrame = pd.DataFrame(columns=['n', 'Silhouette', 'Inertia'])
            resultFrameTrue = pd.DataFrame(columns=['n', 'Mutual_Information', 'Adjusted_Rand'])

            ns = [2,3,4,5,6,7,8,9,10,11,12]
            for j, n in enumerate(ns):
                learner = KMeans(n_clusters=n, random_state=0, n_init = 10)
                learner.fit(all_x_data)
                lab = learner.labels_
                inertia = learner.inertia_
                sil = metrics.silhouette_score(all_x_data, lab)
                mut_info = metrics.completeness_score(y, lab)
                adj_rand = metrics.adjusted_rand_score(y, lab)
                resultFrame.loc[len(resultFrame)] = list([n, sil, inertia])
                resultFrameTrue.loc[len(resultFrameTrue)] = list([n, mut_info, adj_rand])

            # Charts without true labels
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means ")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            ax.set_ylabel('Silhouette Score', color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Inertia Score', color='purple')
            ax.plot('n', 'Silhouette', data=resultFrame, linewidth=2, color = 'blue', linestyle = 'dashed')
            ax2.plot('n', 'Inertia', data=resultFrame, linewidth=2, color = 'purple')
            ax.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig('KMeans_nclusters_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            # Charts with true labels
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means ")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            ax.set_ylabel('Mutual Information Score', color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Adjusted Rand Score', color='purple')
            ax.plot('n', 'Mutual_Information', data=resultFrameTrue, linewidth=2, color='blue', linestyle='dashed')
            ax2.plot('n', 'Adjusted_Rand', data=resultFrameTrue, linewidth=2, color='purple')
            ax.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig('KMeans_nclusters_True_Labels_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                clusters = 6
                f_size = (10, 6)
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                clusters = 3
                f_size = (25,15)
            learner = KMeans(n_clusters=clusters, random_state=0)
            learner.fit(all_x_data)
            lab = learner.labels_
            pd.plotting.scatter_matrix(all_x_data, c=lab, figsize= f_size, hist_kwds={'bins': 20}, s=60, alpha=.8)
            plt.tight_layout()
            plt.savefig('KMeans_Labeled_Scatter_' + name)

    def testEM(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()

            resultFrame = pd.DataFrame(columns=['n', 'Silhouette', 'BIC'])
            resultFrameTrue = pd.DataFrame(columns=['n', 'Mutual_Information', 'Adjusted_Rand'])

            ns = [2,3,4,5,6,7,8,9,10,11,12]
            for j, n in enumerate(ns):
                learner = GaussianMixture(n_components=n, random_state=0, n_init = 10)
                learner.fit(all_x_data)
                lab = learner.predict(all_x_data)
                bic = learner.bic(all_x_data)
                sil = metrics.silhouette_score(all_x_data, lab)
                mut_info = metrics.completeness_score(y, lab)
                adj_rand = metrics.adjusted_rand_score(y, lab)
                resultFrame.loc[len(resultFrame)] = list([n, sil, bic])
                resultFrameTrue.loc[len(resultFrameTrue)] = list([n, mut_info, adj_rand])

            # Charts without true labels
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means ")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            ax.set_ylabel('Silhouette Score', color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('BIC Score', color='purple')
            ax.plot('n', 'Silhouette', data=resultFrame, linewidth=2, color = 'blue', linestyle = 'dashed')
            ax2.plot('n', 'BIC', data=resultFrame, linewidth=2, color = 'purple')
            ax.legend(loc = 'upper left')
            ax2.legend(loc = 'upper right')
            plt.tight_layout()
            plt.savefig('EM_nclusters_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            # Charts with true labels
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means ")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            ax.set_ylabel('Mutual Information Score', color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Adjusted Rand Score', color='purple')
            ax.plot('n', 'Mutual_Information', data=resultFrameTrue, linewidth=2, color='blue', linestyle='dashed')
            ax2.plot('n', 'Adjusted_Rand', data=resultFrameTrue, linewidth=2, color='purple')
            ax.legend(loc = 'upper left')
            ax2.legend(loc = 'upper right')
            plt.tight_layout()
            plt.savefig('EM_nclusters_True_Labels_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                clusters = 6
                f_size = (10, 6)
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                clusters = 3
                f_size = (25,15)
            learner = KMeans(n_clusters=clusters, random_state=0)
            learner.fit(all_x_data)
            lab = learner.labels_
            pd.plotting.scatter_matrix(all_x_data, c=lab, figsize= f_size, hist_kwds={'bins': 20}, s=60, alpha=.8)
            plt.tight_layout()
            plt.savefig('EM_Labeled_Scatter_' + name)

    def testPCA(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ns = [1, 2, 3, 4, 5, 6]
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11]

            resultFrame = pd.DataFrame(columns=['n', 'Explained_Variance', 'Log_Likelihood', 'Reconstruction_Error'])
            for j, n in enumerate(ns):
                learner = PCA(n_components=n, random_state=0)
                transformed_data = learner.fit_transform(all_x_data)
                reconstructed = learner.inverse_transform(transformed_data)
                ssq = np.sum(all_x_data.to_numpy()-reconstructed)**2
                t = learner.explained_variance_
                t2 = learner.explained_variance_ratio_
                att = learner.components_
                if run == 0:
                    eigPoker = learner.explained_variance_
                else:
                    eigTech = learner.explained_variance_
                log_score = learner.score(all_x_data)
                resultFrame.loc[len(resultFrame)] = list([n, t[j], log_score, ssq])
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " PCA ")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_components')
            ax.set_ylabel('Explained_Variance_Ratio Score', color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Reconstruction_Error', color='purple')
            ax.plot('n', 'Explained_Variance', data=resultFrame, linewidth=2, color='blue', linestyle='dashed')
            #ax2.plot('n', 'Log_Likelihood', data=resultFrame, linewidth=2, color='purple')
            ax2.plot('n', 'Reconstruction_Error', data=resultFrame, linewidth=2, color='purple')
            ax.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig('PCA_' + name + '_Results' + '.png')
            plt.show()
            ax.clear()
            plt.clf()

        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ax.set_title(name + " PCA ")
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_xlabel('n_components')
        ax.set_ylabel('Explained_Variance_Ratio Score', color='blue')
        ax.plot(eigTech, label = 'Technical', linewidth=2, color='red')
        ax.plot(eigPoker, label = 'Poker', linewidth=2, color='orange')
        ax.legend()
        plt.tight_layout()
        plt.savefig('PCA_Eigen_' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()

    def testICA(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ns = [1, 2, 3, 4, 5, 6]
                components = 5
                f_size = (10, 6)
                cm = 'Set3'
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                components = 3
                # f_size = (30, 18)
                f_size = (10, 6)
                cm = None

            resultFrame = pd.DataFrame(columns=['n', 'Kurtosis', 'Reconstruction_Error'])
            for j, n in enumerate(ns):
                learner = FastICA(n_components=n, random_state=0)
                transformed_data = learner.fit_transform(all_x_data)
                kurtos = kurtosis(transformed_data,axis=0, fisher=False)
                reconstructed = learner.inverse_transform(transformed_data)
                ssq = np.sum(all_x_data.to_numpy() - reconstructed) ** 2

                # pd.plotting.scatter_matrix(pd.DataFrame(transformed_data), c=y, cmap=cm, figsize=f_size, hist_kwds={'bins': 20}, s=60, alpha=.8)
                # plt.tight_layout()
                # plt.show()
                # plt.clf()

                resultFrame.loc[len(resultFrame)] = list([n, kurtos.mean(), ssq])
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " ICA ")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_components')
            ax.set_ylabel('Kurtosis Score', color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Reconstruction_Error', color='purple')
            ax.plot('n', 'Kurtosis', data=resultFrame, linewidth=2, color='blue', linestyle='dashed')
            ax2.plot('n', 'Reconstruction_Error', data=resultFrame, linewidth=2, color='purple')
            ax.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig('ICA_' + name + '_Results' + '.png')
            plt.show()
            ax.clear()
            plt.clf()

            learner = FastICA(n_components=components, random_state=0)
            transformed_data = pd.DataFrame(learner.fit_transform(all_x_data))
            pd.plotting.scatter_matrix(transformed_data, c=y, cmap=cm, figsize=f_size, hist_kwds={'bins': 20}, s=60, alpha=.8)
            plt.tight_layout()
            plt.savefig('ICA_Scatter_' + name)
            plt.clf()

    def testRCA(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ns = [2, 3, 4, 5,6]
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ns = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " RCA Reconstruction Error")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_components')
            ax.set_ylabel('Reconstruction Sum of Square Error')

            for i in range(20):
                resultFrame = pd.DataFrame(columns=['n', 'Pairwise_Error', 'Reconstruction_Error'])
                for j, n in enumerate(ns):
                    learner = SparseRandomProjection(n_components=n, random_state=i)
                    transformed_data = learner.fit_transform(all_x_data)
                    error = reconstruction_error(learner, all_x_data)
                    resultFrame.loc[len(resultFrame)] = list([n, error, 1])
                ax.plot('n', 'Pairwise_Error', data=resultFrame, linewidth=2)
                # ax.legend()
            plt.tight_layout()
            plt.savefig('RP_' + name + '_Results' + '.png')
            plt.show()
            ax.clear()
            plt.clf()

    def testFA(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ns = [1, 2, 3, 4, 5, 6]
                components = 5
                f_size = (10, 6)
                cm = 'Set3'
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                components = 3
                # f_size = (30, 18)
                f_size = (10, 6)
                cm = None

            resultFrame = pd.DataFrame(columns=['n', 'Average_log_likelihood', 'Last_log_Likelihood'])
            for j, n in enumerate(ns):
                learner = FactorAnalysis(n_components=n,random_state = 0)
                transformed_data = learner.fit_transform(all_x_data)
                ave_log = learner.score(all_x_data)
                last_log = learner.loglike_[-1]
                resultFrame.loc[len(resultFrame)] = list([n, ave_log, last_log])
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " FA ")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_components')
            ax.set_ylabel('Average_log_likelihood Score', color='blue')
            ax2 = ax.twinx()
            ax2.set_ylabel('Last_log_Likelihood', color='purple')
            ax.plot('n', 'Average_log_likelihood', data=resultFrame, linewidth=2, color='blue', linestyle='dashed')
            ax2.plot('n', 'Last_log_Likelihood', data=resultFrame, linewidth=2, color='purple')
            ax.legend()
            ax2.legend()
            plt.tight_layout()
            plt.savefig('FA' + name + '_Results' + '.png')
            plt.show()
            ax.clear()
            plt.clf()

    def testRound2KMeans(self):
        names = ['PCA', 'ICA', 'RCA', 'FA']
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l1 = PCA(n_components=5, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=5, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4 = FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l1 = PCA(n_components=4, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=6, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4= FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            Rs = []
            RsTrue = []
            for j, l in enumerate(ls):
                transformed_data = l.fit_transform(all_x_data)
                resultFrame = pd.DataFrame(columns=['n', 'Silhouette', 'Inertia'])
                resultFrameTrue = pd.DataFrame(columns=['n', 'Mutual_Information', 'Adjusted_Rand'])
                ns = [2,3,4,5,6,7,8,9,10,11,12]
                for k, n in enumerate(ns):
                    learner = KMeans(n_clusters=n, random_state=0, n_init=10)
                    learner.fit(transformed_data)
                    lab = learner.labels_
                    inertia = learner.inertia_
                    sil = metrics.silhouette_score(transformed_data, lab)
                    mut_info = metrics.completeness_score(y, lab)
                    adj_rand = metrics.adjusted_rand_score(y, lab)
                    resultFrame.loc[len(resultFrame)] = list([n, sil, inertia])
                    resultFrameTrue.loc[len(resultFrameTrue)] = list([n, mut_info, adj_rand])
                Rs.append(resultFrame)
                RsTrue.append(resultFrameTrue)

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(Rs):
                ax.set_ylabel('Silhouette Score ' + names[a])
                ax.plot('n', 'Silhouette', data=r, linewidth=2, label='Silhouette ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('KMeans_Reproduced_Silhouette_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(Rs):
                ax.set_ylabel('Inertia Score ' + names[a])
                ax.plot('n', 'Inertia', data=r, linewidth=2, label='Inertia ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('KMeans_Reproduced_Inertia_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(RsTrue):
                ax.set_ylabel('Mutual_Information Score ' + names[a])
                ax.plot('n', 'Mutual_Information', data=r, linewidth=2, label='Mutual_Information ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('KMeans_Reproduced_Mutual_Information_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " K-Means Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(RsTrue):
                ax.set_ylabel('Adjusted_Rand Score ' + names[a])
                ax.plot('n', 'Adjusted_Rand', data=r, linewidth=2, label='Adjusted_Rand ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('KMeans_Reproduced_Adjusted_Rand_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

    def testRound2EM(self):
        names = ['PCA', 'ICA', 'RCA', 'FA']
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l1 = PCA(n_components=5, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=5, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4 = FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l1 = PCA(n_components=4, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=6, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4= FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            Rs = []
            RsTrue = []
            for j, l in enumerate(ls):
                transformed_data = l.fit_transform(all_x_data)
                resultFrame = pd.DataFrame(columns=['n', 'Silhouette', 'BIC'])
                resultFrameTrue = pd.DataFrame(columns=['n', 'Mutual_Information', 'Adjusted_Rand'])
                ns = [2,3,4,5,6,7,8,9,10,11,12]
                for k, n in enumerate(ns):
                    learner = GaussianMixture(n_components=n, random_state=0, n_init=10)
                    learner.fit(transformed_data)
                    lab = learner.predict(transformed_data)
                    BIC = learner.bic(transformed_data)
                    sil = metrics.silhouette_score(transformed_data, lab)
                    mut_info = metrics.completeness_score(y, lab)
                    adj_rand = metrics.adjusted_rand_score(y, lab)
                    resultFrame.loc[len(resultFrame)] = list([n, sil, BIC])
                    resultFrameTrue.loc[len(resultFrameTrue)] = list([n, mut_info, adj_rand])
                Rs.append(resultFrame)
                RsTrue.append(resultFrameTrue)

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " EM Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(Rs):
                ax.set_ylabel('Silhouette Score ' + names[a])
                ax.plot('n', 'Silhouette', data=r, linewidth=2, label='Silhouette ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('EM_Reproduced_Silhouette_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " EM Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(Rs):
                ax.set_ylabel('BIC Score ' + names[a])
                ax.plot('n', 'BIC', data=r, linewidth=2, label='BIC ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('EM_Reproduced_BIC_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " EM Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(RsTrue):
                ax.set_ylabel('Mutual_Information Score ' + names[a])
                ax.plot('n', 'Mutual_Information', data=r, linewidth=2, label='Mutual_Information ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('EM_Reproduced_Mutual_Information_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            ax.set_title(name + " EM Reproduced")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('n_clusters')
            for a, r in enumerate(RsTrue):
                ax.set_ylabel('Adjusted_Rand Score ' + names[a])
                ax.plot('n', 'Adjusted_Rand', data=r, linewidth=2, label='Adjusted_Rand ' + names[a])
                ax.legend()
            plt.tight_layout()
            plt.savefig('EM_Reproduced_Adjusted_Rand_' + name + '_Results' + '.png')
            ax.clear()
            plt.clf()

    def testRound2ICA(self):
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                clusters = 5
                f_size = (10, 6)
            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                clusters = 4
                f_size = (25,15)
            lICA = FastICA(n_components=5, random_state=0)
            transformed_data = lICA.fit_transform(all_x_data)

            learner = KMeans(n_clusters=clusters, random_state=0)
            learner.fit(transformed_data)
            lab = learner.labels_
            pd.plotting.scatter_matrix(all_x_data, c=y, figsize= f_size, hist_kwds={'bins': 20}, s=60, alpha=.8)
            plt.tight_layout()
            plt.savefig('KEAM_ICA_Labeled_Scatter_Real_' + name)

    def testNN(self):
        names = ['No Reduction','PCA', 'ICA', 'RCA', 'FA']
        for run in range(2):
            if run == 0:
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l0 = 0
                ls.append(l0)
                l1 = PCA(n_components=5, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=5, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4 = FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            else:
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l0 = 0
                ls.append(l0)
                l1 = PCA(n_components=4, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=6, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4 = FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            Rs = []
            for j, l in enumerate(ls):
                if j == 0:
                    transformed_data = all_x_data
                else:
                    transformed_data = pd.DataFrame(l.fit_transform(all_x_data))
                resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
                ns = [10,20,30,40,50,60,70,80,90,100]
                for k, n in enumerate(ns):
                    learner = MLPClassifier(hidden_layer_sizes=(n, int(n / 2)), solver='lbfgs', activation='relu', max_iter=500)
                    training_accuracy, test_accuracy = crossValidation(transformed_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                Rs.append(resultFrame)

            fig, ax = plt.subplots(1)
            fig.set_figwidth(15)
            fig.set_figheight(9)
            ax.set_title(name + " Neural Net on Dimensionality Reduction")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('Hidden Layer Size')
            ax.set_ylabel('Accuracy Score')

            cs = ['black','red','blue','purple','green']
            for a, r in enumerate(Rs):
                ax.plot('n', 'Train_Accuracy', data=r, linewidth=2, linestyle='dashed', label='Train_Accuracy ' + names[a], color = cs[a])
                ax.plot('n', 'Test_Accuracy', data=r, linewidth=2, label='Test_Accuracy ' + names[a], color = cs[a])
            ax.legend()
            plt.tight_layout()
            plt.savefig('NN_Dim_Reduct_' + name + '_Results' + '.png')
            plt.show()
            ax.clear()
            plt.clf()

    def testNN2(self):
        names = ['No Reduction','KM PCA', 'KM ICA', 'KM RCA', 'KM FA', 'EM PCA', 'EM ICA', 'EM RCA', 'EM FA']
        for run in range(2):
            if run == 0:
                n_clusters_KM = [6,5,5,5]
                n_clusters_EM = [7,7,7,5]
                all_x_data, all_y_data, name = getPokerData()
                y = (all_y_data - 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l1 = PCA(n_components=5, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=5, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4 = FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            else:
                n_clusters_KM = [2, 4, 5, 2]
                n_clusters_EM = [7, 7, 5, 4]
                all_x_data, all_y_data, name = getTechnicalIndicatorData()
                y = (all_y_data + 1).iloc[:, 0].to_numpy().transpose()
                ls = []
                l1 = PCA(n_components=4, random_state=0)
                ls.append(l1)
                l2 = FastICA(n_components=6, random_state=0)
                ls.append(l2)
                l3 = SparseRandomProjection(n_components=5, random_state=0)
                ls.append(l3)
                l4 = FactorAnalysis(n_components=4, random_state=0)
                ls.append(l4)

            Rs = []
            # Original
            transformed_data = all_x_data
            resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy','Time'])
            ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            start_time = time.time()
            for k, n in enumerate(ns):
                learner = MLPClassifier(hidden_layer_sizes=(n, int(n / 2)), solver='lbfgs', activation='relu', max_iter=200)
                training_accuracy, test_accuracy = crossValidation(transformed_data, all_y_data, learner)
                resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy, time.time() - start_time])
            Rs.append(resultFrame)

            # K Means
            for j, l in enumerate(ls):
                transformed_data = pd.DataFrame(l.fit_transform(all_x_data))
                learner = KMeans(n_clusters=n_clusters_KM[j], random_state=0, n_init=10)
                learner.fit(transformed_data)
                lab = learner.labels_
                transformed_data['labels'] = (lab/lab.max())
                resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy', 'Time'])
                ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                start_time = time.time()
                for k, n in enumerate(ns):
                    learner = MLPClassifier(hidden_layer_sizes=(n, int(n / 2)), solver='lbfgs', activation='relu', max_iter=200)
                    training_accuracy, test_accuracy = crossValidation(transformed_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy, time.time() - start_time])
                Rs.append(resultFrame)

            # EM
            for j, l in enumerate(ls):
                start_time = time.time()
                transformed_data = pd.DataFrame(l.fit_transform(all_x_data))
                learner = GaussianMixture(n_components=n_clusters_EM[j], random_state=0, n_init=10)
                learner.fit(transformed_data)
                lab = learner.predict(transformed_data)
                transformed_data['labels'] = (lab / lab.max())
                resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy', 'Time'])
                ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                for k, n in enumerate(ns):
                    learner = MLPClassifier(hidden_layer_sizes=(n, int(n / 2)), solver='lbfgs', activation='relu',max_iter=200)
                    training_accuracy, test_accuracy = crossValidation(transformed_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy, time.time() - start_time])
                Rs.append(resultFrame)

            fig, ax = plt.subplots(1)
            fig.set_figwidth(15)
            fig.set_figheight(9)
            ax.set_title(name + " Neural Net on Dimensionality Reduction")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('Hidden Layer Size')
            ax.set_ylabel('Accuracy Score')
            cs = ['black','red','blue','purple','green', 'orange', 'yellow', 'pink', 'magenta']
            for a, r in enumerate(Rs):
                ax.plot('n', 'Train_Accuracy', data=r, linewidth=2, linestyle='dashed', label='Train_Accuracy ' + names[a], color = cs[a])
                ax.plot('n', 'Test_Accuracy', data=r, linewidth=2, label='Test_Accuracy ' + names[a], color = cs[a])
            ax.legend()
            plt.tight_layout()
            plt.savefig('NN_Dim_Reduct_5_' + name + '_Results' + '.png')
            plt.show()
            ax.clear()
            plt.clf()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(15)
            fig.set_figheight(9)
            ax.set_title(name + " Neural Net Time")
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_xlabel('Hidden Layer Size')
            ax.set_ylabel('Time')
            for a, r in enumerate(Rs):
                ax.plot('n', 'Time', data=r, linewidth=2, linestyle='dashed', label='Train_Accuracy ' + names[a], color = cs[a])
            ax.legend()
            plt.tight_layout()
            plt.savefig('NN_Dim_Reduct_5_Time_' + name + '_Results' + '.png')
            plt.show()
            ax.clear()
            plt.clf()

class SupervisedLearning(unittest.TestCase):

    def testOne(self):
        all_x_data, all_y_data, name = self.getPokerData()
        maxIter = 100
        # learner = MLPClassifier(hidden_layer_sizes=(100, 50), solver='lbfgs', activation='relu', max_iter=500)

        # learner = mlrose.NeuralNetwork(hidden_nodes=[10, 10], activation='relu',
        #                                algorithm='genetic_alg',
        #                                max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.1,
        #                                early_stopping=True, clip_max=1e+10,
        #                                restarts=0,
        #                                schedule=mlrose.GeomDecay(init_temp=1, decay=0.9, min_temp=.0001),
        #                                pop_size=30, mutation_prob=0.1,
        #                                max_attempts=maxIter, random_state=1)

        # learner = mlrose.NeuralNetwork(hidden_nodes=[80], activation='identity', algorithm='simulated_annealing',
        #                                max_iters=maxIter, bias=True, is_classifier=True, learning_rate=.3,
        #                                early_stopping=True, clip_max=10,
        #                                restarts=0,
        #                                schedule=mlrose.GeomDecay(init_temp=20, decay=.9, min_temp=.0001),
        #                                pop_size=200, mutation_prob=0.1,
        #                                max_attempts=maxIter, random_state=1)

        learner = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu', algorithm='gradient_descent',
                                       max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.0001,
                                       early_stopping=False, clip_max=1e+10,
                                       restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                       max_attempts=maxIter, random_state=111)

        start_time = time.time()
        mytraining_accuracy, mytest_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
        stop_time = time.time()
        run_time = stop_time - start_time
        sys.stdout.write(
            Fore.BLACK + "\n" + ' ' + 'NueralNetwork' + "Run Time: " + Fore.RED + "{:.2} sec".format(run_time) + ' ')
        sys.stdout.write(Fore.BLACK + "\nTrain Accuracy: " + str(mytraining_accuracy) + " Test Accuracy: " + str(
            mytest_accuracy) + ' ')

    def test_GD_NN(self):
        all_x_data, all_y_data, name = self.getPokerData()
        maxIter = 200

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[n], activation='identity', algorithm='gradient_descent',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.0001,
                                           early_stopping=False, clip_max=1e+10,
                                           restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " Gradient Descent ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('N Layers')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_GD' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

        all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
        maxIter = 200

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = ['relu', 'identity', 'sigmoid', 'tanh']
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[30], activation=n, algorithm='gradient_descent',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.0001,
                                           early_stopping=False, clip_max=1e+10,
                                           restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " Gradient Descent ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('N Layers')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_GD_2' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

    def test_RHC_NN(self):
        # best_state, best_fitness, rhc_fitness_curve, rhc_time_curve = mlrose.random_hill_climb(problem, max_attempts=myiter, max_iters=myiter, restarts=0, init_state=None, curve=True, random_state=1)
        # best_state, best_fitness, sa_fitness_curve, sa_time_curve = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=myiter, max_iters=myiter, init_state=None, random_state=1, curve = True)
        # best_state, best_fitness, ga_fitness_curve, ga_time_curve = mlrose.genetic_alg(problem, pop_size=125, mutation_prob=0.05, max_attempts=myiter, max_iters=myiter, curve=True, random_state=1)
        # test = mlrose.NeuralNetwork(hidden_nodes=None,activation='relu',algorithm='random_hill_climb',max_iters=100,bias=True,is_classifier=True,learning_rate=0.1,early_stopping=False,clip_max=1e+10,
        #          restarts=0, schedule=GeomDecay(),pop_size=200,mutation_prob=0.1,max_attempts=10,random_state=None)

        # learner = MLPClassifier(hidden_layer_sizes=(100, 50), solver='lbfgs', activation='relu', max_iter=500)
        # random_hill_climb, simulated_annealing, genetic_alg, gradient_descent

        all_x_data, all_y_data, name = self.getPokerData()
        maxIter = 300

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = [5,10,20,30,40,50,60,70,80,90,100,110,120]
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[n], activation='identity', algorithm='random_hill_climb',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.2,
                                           early_stopping=True, clip_max=1,
                                           restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " random_hill_climb ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('N Layers')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_RHC' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = [100,300,500,700,900,1100]
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[5], activation='identity', algorithm='random_hill_climb',
                                           max_iters=n, bias=True, is_classifier=True, learning_rate=0.2,
                                           early_stopping=True, clip_max=1e+10,
                                           restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                           max_attempts=n, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " random_hill_climb ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_RHC2' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

    def test_SA_NN(self):
        all_x_data, all_y_data, name = self.getPokerData()
        maxIter = 1000

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = [.05,.1,.15,.2,.25,.3,.35,.4]
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[80], activation='identity', algorithm='simulated_annealing',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=n,
                                           early_stopping=True, clip_max=10, restarts=0,
                                           schedule=mlrose.GeomDecay(init_temp=10, decay=.99, min_temp=.1),
                                           pop_size=200, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " simulated_annealing ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_SA' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = [.9,.92,.94,.96,.98,.999]
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[80], activation='identity', algorithm='simulated_annealing',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=.3,
                                           early_stopping=True, clip_max=10,
                                           restarts=0,
                                           schedule=mlrose.GeomDecay(init_temp=20, decay=n, min_temp=.0001),
                                           pop_size=200, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " simulated_annealing ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('Decay Rate')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_SA2' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

    def test_GA_NN(self):
        all_x_data, all_y_data, name = self.getPokerData()
        maxIter = 50

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = [5,10,15,20,25,30]
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[n], activation='identity',
                                           algorithm='genetic_alg',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.1,
                                           early_stopping=True, clip_max=1e+10,
                                           restarts=0,
                                           schedule=mlrose.GeomDecay(init_temp=1, decay=0.9, min_temp=.0001),
                                           pop_size=50, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " genetic_alg ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('N Layers')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_GA' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

        resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
        fig, ax = plt.subplots(1)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ns = [50, 100, 150, 200, 250, 300, 350, 400]
        for j, n in enumerate(ns):
            learner = mlrose.NeuralNetwork(hidden_nodes=[30], activation='identity',
                                           algorithm='genetic_alg',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.1,
                                           early_stopping=False, clip_max=1e+10,
                                           restarts=0,
                                           schedule=mlrose.GeomDecay(init_temp=1, decay=0.9, min_temp=.0001),
                                           pop_size=n, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)
            training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
        ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
        ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
        ax.set_title(name + " genetic_alg ")
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        ax.set_ylim([0, 1])
        ax.set_xlabel('pop_size')
        ax.set_ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig('NN_GA2' + name + '_Results' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

    def test_Times(self):
        maxIter = 1000
        all_x_data, all_y_data, name = self.getPokerData()
        one_hot = OneHotEncoder(categories='auto')
        xTrain, xTest, yTrain, yTest = train_test_split(all_x_data, all_y_data, test_size=0.3, random_state=1)
        y_train_hot = one_hot.fit_transform(yTrain.to_numpy().reshape(-1, 1)).todense()

        # Random Hill Climbing: max attempts, restarts
        start_time = time.time()
        learner = mlrose.NeuralNetwork(hidden_nodes=[5], activation='identity', algorithm='random_hill_climb',
                                       max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.2,
                                       early_stopping=True, clip_max=1e+10,
                                       restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                       max_attempts=maxIter, random_state=1)
        rhc_fitness_curve, rhc_time_curve = learner.fit(xTrain, y_train_hot)


        # Simulated Annealing: schedule (ExpDecay, ArithDecay, GeomDecay), decay parameter of schedule
        learner = mlrose.NeuralNetwork(hidden_nodes=[80], activation='identity', algorithm='simulated_annealing',
                                       max_iters=maxIter, bias=True, is_classifier=True, learning_rate=.3,
                                       early_stopping=True, clip_max=10,
                                       restarts=0,
                                       schedule=mlrose.GeomDecay(init_temp=20, decay=.9, min_temp=.0001),
                                       pop_size=200, mutation_prob=0.1,
                                       max_attempts=maxIter, random_state=1)
        sa_fitness_curve, sa_time_curve = learner.fit(xTrain, y_train_hot)

        # Genetic Algorithm: pop, mutation_prob
        learner = mlrose.NeuralNetwork(hidden_nodes=[30], activation='identity',
                                           algorithm='genetic_alg',
                                           max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.1,
                                           early_stopping=True, clip_max=1e+10,
                                           restarts=0,
                                           schedule=mlrose.GeomDecay(init_temp=1, decay=0.9, min_temp=.0001),
                                           pop_size=50, mutation_prob=0.1,
                                           max_attempts=maxIter, random_state=1)

        ga_fitness_curve, ga_time_curve = learner.fit(xTrain, y_train_hot)

        # Gradient Descent: pop_size, keep_pct
        learner = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu', algorithm='gradient_descent',
                                       max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.0001,
                                       early_stopping=True, clip_max=1e+10,
                                       restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                       max_attempts=maxIter, random_state=1)
        m_fitness_curve, m_time_curve = learner.fit(xTrain, y_train_hot)

        fig, ax = plt.subplots(1)
        fig.set_figwidth(9)
        fig.set_figheight(6)
        ax.plot(rhc_fitness_curve, label='rhc_fitness_curve', linestyle='solid', linewidth=2)
        ax.plot(sa_fitness_curve, label='sa_fitness_curve', linestyle='solid', linewidth=2)
        ax.plot(ga_fitness_curve, label='ga_fitness_curve', linestyle='solid', linewidth=2)
        ax.plot(m_fitness_curve, label='gd_fitness_curve', linestyle='solid', linewidth=2)
        # ax.set_title(name + "Learning Curve")
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness Score (Maximization)')
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        plt.tight_layout()
        plt.savefig('Fitness_poker' +'.png')
        plt.show()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

        fig, ax = plt.subplots(1)
        fig.set_figwidth(12)
        fig.set_figheight(6)
        ax.plot(rhc_time_curve, rhc_fitness_curve, label='rhc_time_curve', linestyle='solid', linewidth=2)
        ax.plot(sa_time_curve, sa_fitness_curve, label='sa_time_curve', linestyle='solid', linewidth=2)
        ax.plot(ga_time_curve, ga_fitness_curve, label='ga_time_curve', linestyle='solid', linewidth=2)
        ax.plot(m_time_curve, m_fitness_curve, label='m_time_curve', linestyle='solid', linewidth=2)
        # ax.set_title(name + "Learning Curve")
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Fitness Score (Maximization)')
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig('Time_poker' + '.png')
        plt.show()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

    def testA2LearningCurves(self):
        maxIter = 300
        all_x_data, all_y_data, name = self.getPokerData()

        fig, ax = plt.subplots(1)
        fig.set_figwidth(12)
        fig.set_figheight(6)

        for run in range(4):
            if run == 0:
                learner = mlrose.NeuralNetwork(hidden_nodes=[5], activation='identity', algorithm='random_hill_climb',
                                               max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.2,
                                               early_stopping=True, clip_max=1e+10,
                                               restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                               max_attempts=maxIter, random_state=1)
                lname = 'Randomized Hill Climbing'
                col = 'green'
            if run == 1:
                learner = mlrose.NeuralNetwork(hidden_nodes=[5], activation='identity',
                                               algorithm='simulated_annealing',
                                               max_iters=maxIter, bias=True, is_classifier=True, learning_rate=.3,
                                               early_stopping=True, clip_max=10,
                                               restarts=0,
                                               schedule=mlrose.GeomDecay(init_temp=20, decay=.9, min_temp=.0001),
                                               pop_size=200, mutation_prob=0.1,
                                               max_attempts=maxIter, random_state=1)
                lname = 'Simulated Annealing'
                col = 'blue'
            if run == 2:
                learner = mlrose.NeuralNetwork(hidden_nodes=[20], activation='identity',
                                               algorithm='genetic_alg',
                                               max_iters=50, bias=True, is_classifier=True, learning_rate=0.1,
                                               early_stopping=True, clip_max=1e+10,
                                               restarts=0,
                                               schedule=mlrose.GeomDecay(init_temp=1, decay=0.9, min_temp=.0001),
                                               pop_size=100, mutation_prob=0.1,
                                               max_attempts=50, random_state=1)
                lname = 'Genetic Algorithm'
                col = 'red'
            if run == 3:
                learner = mlrose.NeuralNetwork(hidden_nodes=[30], activation='relu', algorithm='gradient_descent',
                                               max_iters=maxIter, bias=True, is_classifier=True, learning_rate=0.0001,
                                               early_stopping=True, clip_max=1e+10,
                                               restarts=0, schedule=mlrose.GeomDecay(), pop_size=200, mutation_prob=0.1,
                                               max_attempts=maxIter, random_state=1)
                lname = 'Gradient Descent'
                col = 'orange'


            # mytraining_accuracy, mytest_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
            one_hot = OneHotEncoder(categories='auto')
            all_y_data_hot = one_hot.fit_transform(all_y_data.to_numpy().reshape(-1, 1)).todense()
            train_sizes, training_accuracy, test_accuracy = learning_curve(learner, all_x_data, all_y_data_hot,
                                                                           train_sizes=[.1, .2, .3, .4, .5, .6, .7,
                                                                                        .8, .9, 1], cv=5)
            training_accuracy = training_accuracy.sum(axis=1) / training_accuracy.shape[1]
            test_accuracy = test_accuracy.sum(axis=1) / test_accuracy.shape[1]

            ax.plot(train_sizes, test_accuracy, label=lname + '_Test', linestyle='solid', linewidth=2, color=col)
            ax.plot(train_sizes, training_accuracy, label=lname + 'Train', linestyle='dashed', linewidth=1,
                    color=col)
            ax.set_title(name + "Learning Curve")
            ax.set_xlabel('Training Samples')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            ax.set_ylim([0, 1])
            plt.tight_layout()
        plt.savefig('LearningCurve_' + name + '_Results_' + str(run) + '.png')
        # plt.show()
        ax.clear()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

    def testClockSpeeds(self):
        for myData in range(2):
            if myData == 1:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            for run in range(5):
                if run == 0:
                    learner = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=15,
                                                     random_state=1)
                    lname = 'DecisionTree'
                    col = 'green'
                if run == 1:
                    learner = AdaBoostClassifier(
                        DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=15, random_state=1),
                        n_estimators=50)
                    lname = 'Boost'
                    col = 'blue'
                if run == 2:
                    learner = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute',
                                                   metric='manhattan')
                    lname = 'KNN'
                    col = 'red'
                if run == 3:
                    learner = SVC(kernel='poly', degree=4, gamma='scale')
                    lname = 'SVM'
                    col = 'orange'
                if run == 4:
                    learner = MLPClassifier(hidden_layer_sizes=(100, 50), solver='lbfgs', activation='relu')
                    lname = 'NueralNetwork'
                    col = 'black'

                start_time = time.time()
                mytraining_accuracy, mytest_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                stop_time = time.time()
                run_time = stop_time - start_time

                sys.stdout.write(Fore.BLACK + "\n" + name + ' ' + lname + "Run Time: " + Fore.RED + "{:.2} sec".format(
                    run_time) + ' ')
                # try:
                #     sys.stdout.write(str(learner.n_iter_))
                # except:
                #     test=1
                # try:
                #     sys.stdout.write(str(learner.estimators_))
                # except:
                #     test = 1

    def testLearningCurve(self):
        for myData in range(2):
            if myData == 0:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            fig, ax = plt.subplots(1)
            fig.set_figwidth(12)
            fig.set_figheight(6)

            for run in range(5):
                if run == 0:
                    learner = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=15,
                                                     random_state=1)
                    lname = 'DecisionTree'
                    col = 'green'
                if run == 1:
                    learner = AdaBoostClassifier(
                        DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=15, random_state=1),
                        n_estimators=50)
                    lname = 'Boost'
                    col = 'blue'
                if run == 2:
                    learner = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute',
                                                   metric='manhattan')
                    lname = 'KNN'
                    col = 'red'
                if run == 3:
                    learner = SVC(kernel='poly', degree=4, gamma='scale')
                    lname = 'SVM'
                    col = 'orange'
                if run == 4:
                    learner = MLPClassifier(hidden_layer_sizes=(100, 50), solver='lbfgs', activation='relu')
                    lname = 'NueralNetwork'
                    col = 'black'

                mytraining_accuracy, mytest_accuracy = self.crossValidation(all_x_data, all_y_data, learner)

                train_sizes, training_accuracy, test_accuracy = learning_curve(learner, all_x_data, all_y_data,
                                                                               train_sizes=[.1, .2, .3, .4, .5, .6, .7,
                                                                                            .8, .9, 1], cv=5)
                training_accuracy = training_accuracy.sum(axis=1) / training_accuracy.shape[1]
                test_accuracy = test_accuracy.sum(axis=1) / test_accuracy.shape[1]


                ax.plot(train_sizes, test_accuracy, label=lname + '_Test', linestyle='solid', linewidth=2, color=col)
                ax.plot(train_sizes, training_accuracy, label=lname + 'Train', linestyle='dashed', linewidth=1,
                        color=col)
                ax.set_title(name + "Learning Curve")
                ax.set_xlabel('Training Samples')
                ax.set_ylabel('Accuracy')
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                plt.tight_layout()
            plt.savefig('LearningCurve_' + name + '_Results_' + str(run) + '.png')
            # plt.show()
            ax.clear()
            plt.clf()
            plt.cla()
            fig.clear()
            ax.clear()

    def testDecisionTree(self):
        for run in range(4):
            if run == 0 or run == 1:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            # First Experiment
            if run == 0 or run == 2:
                resultFrame = pd.DataFrame(columns=['p', 'n', 'Train_Accuracy', 'Test_Accuracy'])
                fig, ax = plt.subplots(2)
                fig.set_figwidth(6)
                fig.set_figheight(6)
                ps = ['gini', 'entropy']
                ns = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
                for i, p in enumerate(ps):
                    for j, n in enumerate(ns):
                        learner = DecisionTreeClassifier(criterion=p, min_samples_leaf=n, random_state=1)
                        training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                        resultFrame.loc[len(resultFrame)] = list([p, n, training_accuracy, test_accuracy])
                        # sys.stdout.write(Fore.BLACK + "\nDepth: " + Fore.GREEN + "%s" % learner.get_depth())
                    ax[i].plot('n', 'Train_Accuracy', data=resultFrame[resultFrame['p'] == p], linewidth=1)
                    ax[i].plot('n', 'Test_Accuracy', data=resultFrame[resultFrame['p'] == p], linewidth=1)
                    ax[i].set_title(name + " DT with criterion = %s" % (p))
                    ax[i].set_xlabel('min_samples_leaf')
                    ax[i].set_ylabel('Accuracy')
                    ax[i].legend()
                    ax[i].grid(which='minor', axis='both', linestyle='-')
                    ax[i].grid(which='major', axis='both', linestyle='-')
                    ax[i].set_ylim([0, 1])
                plt.tight_layout()
                plt.savefig('DecisionTree_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                plt.clf()
                plt.cla()
                fig.clear()

            # Second Experiment
            if run == 1 or run == 3:
                resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
                fig, ax = plt.subplots(1)
                fig.set_figwidth(6)
                fig.set_figheight(3)
                ns = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
                for j, n in enumerate(ns):
                    learner = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=n,
                                                     random_state=1)
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " DecisionTree ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Max Depth')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('DecisionTree_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()
                fig.clear()
                ax.clear()

    def testBoostedDecisionTree(self):
        for run in range(4):
            resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            if run == 0 or run == 1:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            # First Experiment
            if run == 0 or run == 2:
                ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                for j, n in enumerate(ns):
                    learner = AdaBoostClassifier(
                        DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, random_state=1, max_depth=15),
                        n_estimators=n)
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " Boost ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('n estimators')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('Boost_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            # Second Experiment
            if run == 1 or run == 3:
                ns = [5, 10, 15, 20, 25, 30]
                for j, n in enumerate(ns):
                    learner = AdaBoostClassifier(
                        DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=n, random_state=1),
                        n_estimators=50)
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " Boost ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Max Depth')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('Boost_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            fig.clear()
            ax.clear()

    def testNeuralNet(self):
        names = ['PCA', 'ICA', 'RCA', 'FA']
        for run in range(2):
            resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            if run == 0 or run == 1:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            # First Experiment
            if run == 0 or run == 2:
                ns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                for j, n in enumerate(ns):
                    learner = MLPClassifier(hidden_layer_sizes=(n, int(n / 2)), solver='lbfgs', activation='relu',
                                            max_iter=500)
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1, label = 'Train_Accuracy '+ names[j])
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " NN ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Layer size')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('NN_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            # Second Experiment
            if run == 1 or run == 3:
                ns = ['identity', 'tanh', 'relu', 'logistic']
                for j, n in enumerate(ns):
                    learner = MLPClassifier(hidden_layer_sizes=(100, 50), solver='lbfgs', activation=n, max_iter=500)
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " NN ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Activation')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('NN_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            fig.clear()
            ax.clear()

    def testSVM(self):
        # learner = linear_model.SGDClassifier(alpha=1e-9, random_state=0)
        for run in range(4):
            resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            if run == 0 or run == 1:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            # First Experiment
            if run == 0 or run == 2:
                ns = ['rbf', 'linear', 'poly', 'sigmoid']
                for j, n in enumerate(ns):
                    learner = SVC(kernel=n, gamma='scale')
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " SVM ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Kernel Fucntions')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('SVM_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            # possible: C - penaluty parameter

            # Second Experiment
            if run == 1 or run == 3:
                ns = [1, 2, 3, 4]
                for j, n in enumerate(ns):
                    learner = SVC(kernel='poly', degree=n, gamma='scale')
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " SVM ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('SVM')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('SVM_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            fig.clear()
            ax.clear()

    def testKNN(self):
        for run in range(4):
            resultFrame = pd.DataFrame(columns=['n', 'Train_Accuracy', 'Test_Accuracy'])
            fig, ax = plt.subplots(1)
            fig.set_figwidth(6)
            fig.set_figheight(3)
            if run == 0 or run == 1:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            # First Experiment
            if run == 0 or run == 2:
                ns = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
                for j, n in enumerate(ns):
                    learner = KNeighborsClassifier(n_neighbors=n, weights='uniform', algorithm='brute')
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " KNN ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Number of Neighbors (n)')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('KNN_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            # Second Experiment
            if run == 1 or run == 3:

                ns = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                for j, n in enumerate(ns):
                    learner = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute', metric=n)
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
                ax.plot('n', 'Test_Accuracy', data=resultFrame, linewidth=1)
                ax.set_title(name + " KNN ")
                ax.legend()
                ax.grid(which='minor', axis='both', linestyle='-')
                ax.grid(which='major', axis='both', linestyle='-')
                ax.set_ylim([0, 1])
                ax.set_xlabel('Metric')
                ax.set_ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('KNN_' + name + '_Results' + str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            fig.clear()
            ax.clear()

class UnsupervisedLearning(unittest.TestCase):

    # Notes for mlrose
    # Possible fitness functions:
    # mlrose.OneMax: np.array([0, 1, 0, 1, 1, 1, 1])
    # mlrose.FlipFlop: np.array([0, 1, 0, 1, 1, 1, 1])
    # mlrose.FourPeaks: np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    # mlrose.SixPeaks: np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1])
    # mlrose.ContinuousPeaks: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1])
    # mlrose.Knapsack: weights = np.array([1, 0, 2, 1, 0]) + several parameters
    # mlrose.TravelingSales: np.array([0, 1, 4, 3, 2]) + coords and dists
    # mlrose.Queens: np.array([1, 4, 1, 3, 5, 5, 2, 7])
    # mlrose.MaxColor: np.array([0, 1, 0, 1, 1]) + edges
    # mlrose.CustomFitness:
    # def cust_fn(state, c): return c * np.sum(state)
    # kwargs = {'c': 10}
    # fitness = mlrose.CustomFitness(cust_fn, **kwargs)

    # Possible optimization problem objects:
    # mlrose.DiscreteOpt(OptProb): for discrete problems
    # mlrose.ContinuousOpt(OptProb): for continuous problems
    # mlrose.TSPOpt(DiscreteOpt): for traveling salesman problems

    # procedure:
    # Final analysis will compare optimal parameters fitness to iteration for each dataset
    # Dataset
    # fitness vs iteration chart with 4 lines representing each algorithm.
    # analysis per dataset will compare fitness to hyperparameters (2 for each))
    # 4 algorithms
    # 2 hyperparameter charts with 4 lines each representing one of the datasets

    def testCombined(self):
        myiter = 200
        population_size = 100

        flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(population_size)

        problem = oneMaxProblem
        t = 'oneMax TIme'

        # Random Hill Climbing: max attempts, restarts
        start_time = time.time()
        best_state, best_fitness, rhc_fitness_curve, rhc_time_curve = mlrose.random_hill_climb(problem,
                                                                                               max_attempts=myiter,
                                                                                               max_iters=myiter,
                                                                                               restarts=0,
                                                                                               init_state=None,
                                                                                               curve=True,
                                                                                               random_state=1)
        print(
            '\nrandom_hill_climb: \nBest State: %s \nBest Fitness: %s\nfitness_curve: %s\nTime_Curve: %s\nRun Time: %s' % (
            str(best_state), str(best_fitness), str(rhc_fitness_curve), str(rhc_time_curve),
            str(time.time() - start_time)))

        # Simulated Annealing: schedule (ExpDecay, ArithDecay, GeomDecay), decay parameter of schedule
        start_time = time.time()
        schedule = mlrose.GeomDecay(init_temp=1, decay=0.9, min_temp=.0001)
        best_state, best_fitness, sa_fitness_curve, sa_time_curve = mlrose.simulated_annealing(problem,
                                                                                               schedule=schedule,
                                                                                               max_attempts=myiter,
                                                                                               max_iters=myiter,
                                                                                               init_state=None,
                                                                                               random_state=1,
                                                                                               curve=True)
        print(
            '\nsimulated_annealing: \nBest State: %s \nBest Fitness: %s\nfitness_curve: %s\nTime_Curve: %s\nRun Time: %s' % (
            str(best_state), str(best_fitness), str(sa_fitness_curve), str(sa_time_curve),
            str(time.time() - start_time)))

        # Genetic Algorithm: pop, mutation_prob
        start_time = time.time()
        best_state, best_fitness, ga_fitness_curve, ga_time_curve = mlrose.genetic_alg(problem, pop_size=125,
                                                                                       mutation_prob=0.05,
                                                                                       max_attempts=myiter,
                                                                                       max_iters=myiter, curve=True,
                                                                                       random_state=1)
        print('\ngenetic_alg: \nBest State: %s \nBest Fitness: %s\nfitness_curve: %s\nTime_Curve: %s\nRun Time: %s' % (
        str(best_state), str(best_fitness), str(ga_fitness_curve), str(ga_time_curve), str(time.time() - start_time)))

        # Mimic: pop_size, keep_pct
        start_time = time.time()
        best_state, best_fitness, m_fitness_curve, m_time_curve = mlrose.mimic(problem, pop_size=200, keep_pct=0.2,
                                                                               max_attempts=myiter, max_iters=myiter,
                                                                               curve=True, random_state=1)
        print('\nmimic: \nBest State: %s \nBest Fitness: %s\nfitness_curve: %s\nTime_Curve: %s\nRun Time: %s' % (
        str(best_state), str(best_fitness), str(m_fitness_curve), str(m_time_curve), str(time.time() - start_time)))

        # fig, ax = plt.subplots(1)
        # fig.set_figwidth(9)
        # fig.set_figheight(6)
        # ax.plot(rhc_fitness_curve, label='rhc_fitness_curve', linestyle='solid', linewidth=2)
        # ax.plot(sa_fitness_curve, label='sa_fitness_curve', linestyle='solid', linewidth=2)
        # ax.plot(ga_fitness_curve, label='ga_fitness_curve', linestyle='solid', linewidth=2)
        # ax.plot(m_fitness_curve, label='m_fitness_curve', linestyle='solid', linewidth=2)
        # # ax.set_title(name + "Learning Curve")
        # ax.set_xlabel('Iteration')
        # ax.set_ylabel('Fitness Score (Maximization)')
        # ax.legend()
        # ax.grid(which='minor', axis='both', linestyle='-')
        # ax.grid(which='major', axis='both', linestyle='-')
        # plt.tight_layout()
        # plt.savefig('Fitness' +'.png')
        # plt.show()
        # plt.clf()
        # plt.cla()
        # fig.clear()
        # ax.clear()

        fig, ax = plt.subplots(1)
        fig.set_figwidth(12)
        fig.set_figheight(6)
        ax.plot(rhc_time_curve, rhc_fitness_curve, label='rhc_time_curve', linestyle='solid', linewidth=2)
        ax.plot(sa_time_curve, sa_fitness_curve, label='sa_time_curve', linestyle='solid', linewidth=2)
        ax.plot(ga_time_curve, ga_fitness_curve, label='ga_time_curve', linestyle='solid', linewidth=2)
        ax.plot(m_time_curve, m_fitness_curve, label='m_time_curve', linestyle='solid', linewidth=2)
        # ax.set_title(name + "Learning Curve")
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Fitness Score (Maximization)')
        ax.legend()
        ax.grid(which='minor', axis='both', linestyle='-')
        ax.grid(which='major', axis='both', linestyle='-')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(t + '.png')
        plt.show()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

    def testRandomHillCLimbQuad(self):
        # Random Hill Climbing: iterations, restarts
        population_sizes = [50, 100, 150, 200]
        myiter = 5000
        algorithm = 'random_hill_climb'

        # Intiatiate Plots
        fig, ax = plt.subplots(2, 2)
        fig.set_figwidth(12)
        fig.set_figheight(6)

        # loop through  population sizes
        for index, size in enumerate(population_sizes):
            if index == 0:
                i = 0
                j = 0
            elif index == 1:
                i = 0
                j = 1
            elif index == 2:
                i = 1
                j = 0
            else:
                i = 1
                j = 1

            # Gather problems based on input data
            flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

            # Loop through problems
            for p in range(3):
                start_time = time.time()
                if p == 0:
                    name = "flipFlopProblem size: " + str(size)
                    problem = flipFlopProblem
                    lstyle = 'solid'
                    col = 'purple'
                elif p == 1:
                    name = "fourPearksProblem size: " + str(size)
                    problem = fourPearksProblem
                    lstyle = 'dashed'
                    col = 'blue'
                else:
                    name = "oneMaxProblem size: " + str(size)
                    problem = oneMaxProblem
                    lstyle = 'dotted'
                    col = 'red'

                # Perform algorithm evaluation and store data
                best_state, best_fitness, rhc_fitness_curve, rhc_time_curve = mlrose.random_hill_climb(problem,
                                                                                                       max_attempts=myiter,
                                                                                                       max_iters=myiter,
                                                                                                       restarts=0,
                                                                                                       init_state=None,
                                                                                                       curve=True,
                                                                                                       random_state=1)
                print('\n' + algorithm + ' ' + name + ': \nBest State: %s \nBest Fitness: %s\nRun Time: %s' % (
                str(best_state), str(best_fitness), str(time.time() - start_time)))

                # Plot the data
                ax[i, j].plot(rhc_fitness_curve, label=name, linestyle=lstyle, linewidth=1, color=col)
                ax[i, j].set_xlabel('Iterations')
                ax[i, j].set_ylabel('Fitness Score (Maximization)')
                ax[i, j].legend(fancybox=True, framealpha=0.5)
                ax[i, j].grid(which='minor', axis='both', linestyle='-')
                ax[i, j].grid(which='major', axis='both', linestyle='-')
                ax[i, j].set_ylim([0, 200])

        # Clean up Plots
        plt.tight_layout()
        plt.savefig(algorithm + ' ' + 'quad opt' + '.png')
        plt.show()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

    def testSimulatedAnnealingQuad(self):
        # Random Hill Climbing: iterations, restarts
        population_sizes = [50, 100, 150, 200]
        myiter = 5000
        algorithm = 'Simulated Annealing'

        # Intiatiate Plots
        fig, ax = plt.subplots(2, 2)
        fig.set_figwidth(12)
        fig.set_figheight(6)

        # loop through  population sizes
        for index, size in enumerate(population_sizes):
            if index == 0:
                i = 0
                j = 0
            elif index == 1:
                i = 0
                j = 1
            elif index == 2:
                i = 1
                j = 0
            else:
                i = 1
                j = 1

            # Gather problems based on input data
            flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

            # Loop through problems
            for p in range(3):
                start_time = time.time()
                if p == 0:
                    name = "flipFlopProblem size: " + str(size)
                    problem = flipFlopProblem
                    lstyle = 'solid'
                    col = 'purple'
                elif p == 1:
                    name = "fourPearksProblem size: " + str(size)
                    problem = fourPearksProblem
                    lstyle = 'dashed'
                    col = 'blue'
                else:
                    name = "oneMaxProblem size: " + str(size)
                    problem = oneMaxProblem
                    lstyle = 'dotted'
                    col = 'red'

                # Perform algorithm evaluation and store data
                schedule = mlrose.GeomDecay(init_temp=1, decay=0.99, min_temp=.0001)
                best_state, best_fitness, sa_fitness_curve, sa_time_curve = mlrose.simulated_annealing(problem,
                                                                                                       schedule=schedule,
                                                                                                       max_attempts=myiter,
                                                                                                       max_iters=myiter,
                                                                                                       init_state=None,
                                                                                                       random_state=1,
                                                                                                       curve=True)
                print('\n' + algorithm + ' ' + name + ': \nBest State: %s \nBest Fitness: %s\nRun Time: %s' % (
                    str(best_state), str(best_fitness), str(time.time() - start_time)))

                # Plot the data
                ax[i, j].plot(sa_fitness_curve, label=name, linestyle=lstyle, linewidth=1, color=col)
                ax[i, j].set_xlabel('Iterations')
                ax[i, j].set_ylabel('Fitness Score (Maximization)')
                ax[i, j].legend(fancybox=True, framealpha=0.5)
                ax[i, j].grid(which='minor', axis='both', linestyle='-')
                ax[i, j].grid(which='major', axis='both', linestyle='-')
                ax[i, j].set_ylim([0, 200])

        # Clean up Plots
        plt.tight_layout()
        plt.savefig(algorithm + ' ' + 'quad opt' + '.png')
        plt.show()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

    def testSimulatedAnnealingParameters(self):

        size = 100
        myiter = 100
        fig, ax = plt.subplots(1)
        fig.set_figwidth(7)
        fig.set_figheight(4.5)

        # Gather problems based on input data
        flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

        # Loop through problems
        for p in range(3):
            start_time = time.time()
            if p == 0:
                name = "flipFlopProblem size: " + str(size)
                problem = flipFlopProblem
                lstyle = 'solid'
                col = 'purple'
            elif p == 1:
                name = "fourPearksProblem size: " + str(size)
                problem = fourPearksProblem
                lstyle = 'dashed'
                col = 'blue'
            else:
                name = "oneMaxProblem size: " + str(size)
                problem = oneMaxProblem
                lstyle = 'dotted'
                col = 'red'
            resultFrame = pd.DataFrame(columns=['t', 'Fitness_Score'])
            x = np.linspace(.01, 5, 100)
            for i, t in enumerate(x):
                schedule = mlrose.GeomDecay(init_temp=t, decay=0.99, min_temp=.0001)
                best_state, best_fitness, sa_fitness_curve, sa_time_curve = mlrose.simulated_annealing(problem,
                                                                                                       schedule=schedule,
                                                                                                       max_attempts=myiter,
                                                                                                       max_iters=myiter,
                                                                                                       init_state=None,
                                                                                                       random_state=1,
                                                                                                       curve=True)
                resultFrame.loc[len(resultFrame)] = list([t, best_fitness])
            ax.plot(resultFrame['t'], resultFrame['Fitness_Score'], label=name, linestyle=lstyle, linewidth=2,
                    color=col)
            ax.set_title('Fitness_Score versus Intial Temperature')
            ax.legend()
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            # ax.set_ylim([0, 1])
            ax.set_xlabel('Intial Temperature')
            ax.set_ylabel('Fitness Score')
        plt.tight_layout()
        # plt.xticks(x)
        plt.savefig('Simulated Annealing' + ' ' + 'temps' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

        # Experiement 2
        fig, ax = plt.subplots(1)
        fig.set_figwidth(7)
        fig.set_figheight(4.5)

        # Gather problems based on input data
        flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

        # Loop through problems
        for p in range(3):
            start_time = time.time()
            if p == 0:
                name = "flipFlopProblem size: " + str(size)
                problem = flipFlopProblem
                lstyle = 'solid'
                col = 'purple'
            elif p == 1:
                name = "fourPearksProblem size: " + str(size)
                problem = fourPearksProblem
                lstyle = 'dashed'
                col = 'blue'
            else:
                name = "oneMaxProblem size: " + str(size)
                problem = oneMaxProblem
                lstyle = 'dotted'
                col = 'red'
            resultFrame = pd.DataFrame(columns=['t', 'Fitness_Score'])
            x = np.linspace(.8, 1, 100)
            for i, t in enumerate(x):
                schedule = mlrose.GeomDecay(init_temp=1, decay=t, min_temp=.0001)
                best_state, best_fitness, sa_fitness_curve, sa_time_curve = mlrose.simulated_annealing(problem,
                                                                                                       schedule=schedule,
                                                                                                       max_attempts=myiter,
                                                                                                       max_iters=myiter,
                                                                                                       init_state=None,
                                                                                                       random_state=1,
                                                                                                       curve=True)
                resultFrame.loc[len(resultFrame)] = list([t, best_fitness])
            ax.plot(resultFrame['t'], resultFrame['Fitness_Score'], label=name, linestyle=lstyle, linewidth=2,
                    color=col)
            ax.set_title('Fitness_Score versus Geometric Decay')
            ax.legend()
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            # ax.set_ylim([0, 1])
            ax.set_xlabel('Geometric Decay')
            ax.set_ylabel('Fitness Score')
        plt.tight_layout()
        # plt.xticks(x)
        plt.savefig('Simulated Annealing' + ' ' + 'decay' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

    def testGeneticAlgorithmQuad(self):
        # Random Hill Climbing: iterations, restarts
        population_sizes = [50, 100, 150, 200]
        myiter = 1000
        algorithm = 'Genetic Algorithm'

        # Intiatiate Plots
        fig, ax = plt.subplots(2, 2)
        fig.set_figwidth(12)
        fig.set_figheight(6)

        # loop through  population sizes
        for index, size in enumerate(population_sizes):
            if index == 0:
                i = 0
                j = 0
            elif index == 1:
                i = 0
                j = 1
            elif index == 2:
                i = 1
                j = 0
            else:
                i = 1
                j = 1

            # Gather problems based on input data
            flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

            # Loop through problems
            for p in range(3):
                start_time = time.time()
                if p == 0:
                    name = "flipFlopProblem size: " + str(size)
                    problem = flipFlopProblem
                    lstyle = 'solid'
                    col = 'purple'
                elif p == 1:
                    name = "fourPearksProblem size: " + str(size)
                    problem = fourPearksProblem
                    lstyle = 'dashed'
                    col = 'blue'
                else:
                    name = "oneMaxProblem size: " + str(size)
                    problem = oneMaxProblem
                    lstyle = 'dotted'
                    col = 'red'

                # Perform algorithm evaluation and store data
                best_state, best_fitness, ga_fitness_curve, ga_time_curve = mlrose.genetic_alg(problem, pop_size=100,
                                                                                               mutation_prob=0.05,
                                                                                               max_attempts=myiter,
                                                                                               max_iters=myiter,
                                                                                               curve=True,
                                                                                               random_state=1)
                print(
                    '\ngenetic_alg: \nBest State: %s \nBest Fitness: %s\nfitness_curve: %s\nTime_Curve: %s\nRun Time: %s' % (
                    str(best_state), str(best_fitness), str(ga_fitness_curve), str(ga_time_curve),
                    str(time.time() - start_time)))

                # Plot the data
                ax[i, j].plot(ga_fitness_curve, label=name, linestyle=lstyle, linewidth=1, color=col)
                ax[i, j].set_xlabel('Iterations')
                ax[i, j].set_ylabel('Fitness Score (Maximization)')
                ax[i, j].legend(fancybox=True, framealpha=0.5)
                ax[i, j].grid(which='minor', axis='both', linestyle='-')
                ax[i, j].grid(which='major', axis='both', linestyle='-')
                ax[i, j].set_ylim([0, 200])

        # Clean up Plots
        plt.tight_layout()
        plt.savefig(algorithm + ' ' + 'quad opt' + '.png')
        plt.show()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

    def testGeneticAlgorithmParameters(self):
        size = 100
        myiter = 100
        fig, ax = plt.subplots(1)
        fig.set_figwidth(7)
        fig.set_figheight(4.5)

        # Gather problems based on input data
        flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

        # Loop through problems
        for p in range(3):
            start_time = time.time()
            if p == 0:
                name = "flipFlopProblem size: " + str(size)
                problem = flipFlopProblem
                lstyle = 'solid'
                col = 'purple'
            elif p == 1:
                name = "fourPearksProblem size: " + str(size)
                problem = fourPearksProblem
                lstyle = 'dashed'
                col = 'blue'
            else:
                name = "oneMaxProblem size: " + str(size)
                problem = oneMaxProblem
                lstyle = 'dotted'
                col = 'red'
            resultFrame = pd.DataFrame(columns=['t', 'Fitness_Score'])
            x = np.linspace(10, 400, 10)
            for i, t in enumerate(x):
                best_state, best_fitness, ga_fitness_curve, ga_time_curve = mlrose.genetic_alg(problem, pop_size=int(t),
                                                                                               mutation_prob=0.05,
                                                                                               max_attempts=myiter,
                                                                                               max_iters=myiter,
                                                                                               curve=True,
                                                                                               random_state=1)
                resultFrame.loc[len(resultFrame)] = list([t, best_fitness])
            ax.plot(resultFrame['t'], resultFrame['Fitness_Score'], label=name, linestyle=lstyle, linewidth=2,
                    color=col)
            ax.set_title('Fitness_Score versus Population Size')
            ax.legend()
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            # ax.set_ylim([0, 1])
            ax.set_xlabel('pop_size')
            ax.set_ylabel('Fitness Score')
        plt.tight_layout()
        # plt.xticks(x)
        plt.savefig('Genetic Algorithm' + ' ' + 'pop_size' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

        # Experiement 2
        fig, ax = plt.subplots(1)
        fig.set_figwidth(7)
        fig.set_figheight(4.5)

        # Gather problems based on input data
        flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

        # Loop through problems
        for p in range(3):
            start_time = time.time()
            if p == 0:
                name = "flipFlopProblem size: " + str(size)
                problem = flipFlopProblem
                lstyle = 'solid'
                col = 'purple'
            elif p == 1:
                name = "fourPearksProblem size: " + str(size)
                problem = fourPearksProblem
                lstyle = 'dashed'
                col = 'blue'
            else:
                name = "oneMaxProblem size: " + str(size)
                problem = oneMaxProblem
                lstyle = 'dotted'
                col = 'red'
            resultFrame = pd.DataFrame(columns=['t', 'Fitness_Score'])
            x = np.linspace(.01, .2, 10)
            for i, t in enumerate(x):
                best_state, best_fitness, ga_fitness_curve, ga_time_curve = mlrose.genetic_alg(problem, pop_size=100,
                                                                                               mutation_prob=t,
                                                                                               max_attempts=myiter,
                                                                                               max_iters=myiter,
                                                                                               curve=True,
                                                                                               random_state=1)
                resultFrame.loc[len(resultFrame)] = list([t, best_fitness])
            ax.plot(resultFrame['t'], resultFrame['Fitness_Score'], label=name, linestyle=lstyle, linewidth=2,
                    color=col)
            ax.set_title('Fitness_Score versus Mutation Probability')
            ax.legend()
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            # ax.set_ylim([0, 1])
            ax.set_xlabel('mutation_prob')
            ax.set_ylabel('Fitness Score')
        plt.tight_layout()
        # plt.xticks(x)
        plt.savefig('Genetic Algorithm' + ' ' + 'mutation_prob' + '.png')
        plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

    def testMimicQuad(self):
        # Random Hill Climbing: iterations, restarts
        population_sizes = [50, 100, 150, 200]
        myiter = 100
        algorithm = 'Mimic'

        # Intiatiate Plots
        fig, ax = plt.subplots(2, 2)
        fig.set_figwidth(12)
        fig.set_figheight(6)

        # loop through  population sizes
        for index, size in enumerate(population_sizes):
            if index == 0:
                i = 0
                j = 0
            elif index == 1:
                i = 0
                j = 1
            elif index == 2:
                i = 1
                j = 0
            else:
                i = 1
                j = 1

            # Gather problems based on input data
            flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

            # Loop through problems
            for p in range(3):
                start_time = time.time()
                if p == 0:
                    name = "flipFlopProblem size: " + str(size)
                    problem = flipFlopProblem
                    lstyle = 'solid'
                    col = 'purple'
                elif p == 1:
                    name = "fourPearksProblem size: " + str(size)
                    problem = fourPearksProblem
                    lstyle = 'dashed'
                    col = 'blue'
                else:
                    name = "oneMaxProblem size: " + str(size)
                    problem = oneMaxProblem
                    lstyle = 'dotted'
                    col = 'red'

                # Perform algorithm evaluation and store data
                best_state, best_fitness, m_fitness_curve, m_time_curve = mlrose.mimic(problem, pop_size=200,
                                                                                       keep_pct=0.2,
                                                                                       max_attempts=myiter,
                                                                                       max_iters=myiter, curve=True,
                                                                                       random_state=1)

                # Plot the data
                ax[i, j].plot(m_fitness_curve, label=name, linestyle=lstyle, linewidth=1, color=col)
                ax[i, j].set_xlabel('Iterations')
                ax[i, j].set_ylabel('Fitness Score (Maximization)')
                ax[i, j].legend(fancybox=True, framealpha=0.5)
                ax[i, j].grid(which='minor', axis='both', linestyle='-')
                ax[i, j].grid(which='major', axis='both', linestyle='-')
                ax[i, j].set_ylim([0, 200])

        # Clean up Plots
        plt.tight_layout()
        plt.savefig(algorithm + ' ' + 'quad opt' + '.png')
        plt.show()
        plt.clf()
        plt.cla()
        fig.clear()
        ax.clear()

    def testMimicParameters(self):
        size = 100
        myiter = 100
        fig, ax = plt.subplots(1)
        fig.set_figwidth(7)
        fig.set_figheight(4.5)

        # Gather problems based on input data
        flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

        # Loop through problems
        for p in range(3):
            start_time = time.time()
            if p == 0:
                name = "flipFlopProblem size: " + str(size)
                problem = flipFlopProblem
                lstyle = 'solid'
                col = 'purple'
            elif p == 1:
                name = "fourPearksProblem size: " + str(size)
                problem = fourPearksProblem
                lstyle = 'dashed'
                col = 'blue'
            else:
                name = "oneMaxProblem size: " + str(size)
                problem = oneMaxProblem
                lstyle = 'dotted'
                col = 'red'
            resultFrame = pd.DataFrame(columns=['t', 'Fitness_Score'])
            x = np.linspace(10, 400, 10)
            for i, t in enumerate(x):
                best_state, best_fitness, m_fitness_curve, m_time_curve = mlrose.mimic(problem, pop_size=int(t),
                                                                                       keep_pct=0.2,
                                                                                       max_attempts=myiter,
                                                                                       max_iters=myiter, curve=True,
                                                                                       random_state=1)
                resultFrame.loc[len(resultFrame)] = list([t, best_fitness])
            ax.plot(resultFrame['t'], resultFrame['Fitness_Score'], label=name, linestyle=lstyle, linewidth=2,
                    color=col)
            ax.set_title('Fitness_Score versus Population Size')
            ax.legend()
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            # ax.set_ylim([0, 1])
            ax.set_xlabel('pop_size')
            ax.set_ylabel('Fitness Score')
        plt.tight_layout()
        # plt.xticks(x)
        plt.savefig('Mimic' + ' ' + 'pop_size' + '.png')
        # plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

        # Experiement 2
        fig, ax = plt.subplots(1)
        fig.set_figwidth(7)
        fig.set_figheight(4.5)

        # Gather problems based on input data
        flipFlopProblem, fourPearksProblem, oneMaxProblem = self.getProblems(size)

        # Loop through problems
        for p in range(3):
            start_time = time.time()
            if p == 0:
                name = "flipFlopProblem size: " + str(size)
                problem = flipFlopProblem
                lstyle = 'solid'
                col = 'purple'
            elif p == 1:
                name = "fourPearksProblem size: " + str(size)
                problem = fourPearksProblem
                lstyle = 'dashed'
                col = 'blue'
            else:
                name = "oneMaxProblem size: " + str(size)
                problem = oneMaxProblem
                lstyle = 'dotted'
                col = 'red'
            resultFrame = pd.DataFrame(columns=['t', 'Fitness_Score'])
            x = np.linspace(.01, .5, 10)
            for i, t in enumerate(x):
                best_state, best_fitness, m_fitness_curve, m_time_curve = mlrose.mimic(problem, pop_size=200,
                                                                                       keep_pct=t, max_attempts=myiter,
                                                                                       max_iters=myiter, curve=True,
                                                                                       random_state=1)
                resultFrame.loc[len(resultFrame)] = list([t, best_fitness])
            ax.plot(resultFrame['t'], resultFrame['Fitness_Score'], label=name, linestyle=lstyle, linewidth=2,
                    color=col)
            ax.set_title('Fitness_Score versus Keep Percentage')
            ax.legend()
            ax.grid(which='minor', axis='both', linestyle='-')
            ax.grid(which='major', axis='both', linestyle='-')
            # ax.set_ylim([0, 1])
            ax.set_xlabel('keep_pct')
            ax.set_ylabel('Fitness Score')
        plt.tight_layout()
        # plt.xticks(x)
        plt.savefig('Mimic' + ' ' + 'keep_pct' + '.png')
        # plt.show()
        ax.clear()
        plt.clf()
        plt.cla()

    def getProblems(self, population_size):
        fitness1 = mlrose.FlipFlop()  # highlight advantages of Miic
        flipFlopProblem = mlrose.DiscreteOpt(length=population_size, fitness_fn=fitness1, maximize=True, max_val=2)

        fitness2 = mlrose.FourPeaks()  # highlight advantages of Genetic Algorithm
        fourPearksProblem = mlrose.DiscreteOpt(length=population_size, fitness_fn=fitness2, maximize=True, max_val=2)

        fitness3 = mlrose.OneMax()  # highlight advantages of Simulated Annealling and RHC
        oneMaxProblem = mlrose.DiscreteOpt(length=population_size, fitness_fn=fitness3, maximize=True, max_val=2)

        return flipFlopProblem, fourPearksProblem, oneMaxProblem