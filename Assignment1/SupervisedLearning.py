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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
import time


#todo: learn more about sklearn Validation Curve
#todo: learn more about sklearn GridSearchCV
#todo: learn more about sklearn pipeline
#todo: learn more about sklearn make scorer
#todo: learn more about sklean learning_curve
#todo: learn more about sklearn compute_sample_weight
#todo: learn more about sklean SelectFromModel

# Notes for ML4T:
# Adaboost takes a long time to run,  Maybe instead consider ensemble.  Currently not using ensemble.
# KNN has performed surprisingly well with low numbers of leafs
# I need to automate searching through hyperparameters
# Use cross validation instead of the 6 chart methods
# The performance measure (score) does not have to be return based on trades.  I can use squared error instead.
    # keep in mind though that the output do not match 1 to 1.  Prediction is much lower.
    # Consider using accuracy score from sklearn.

print("Hello")

class SupervisedLearning(unittest.TestCase):

    def testClockSpeeds(self):
        for myData in range(2):
            if myData == 0:
                all_x_data, all_y_data, name = self.getTechnicalIndicatorData()
            else:
                all_x_data, all_y_data, name = self.getPokerData()

            for run in range(5):
                if run == 0:
                    learner = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth = 15, random_state=1)
                    lname = 'DecisionTree'
                    col = 'green'
                if run == 1:
                    learner = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=15, random_state=1), n_estimators=50)
                    lname = 'Boost'
                    col = 'blue'
                if run == 2:
                    learner = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute', metric='manhattan')
                    lname = 'KNN'
                    col = 'red'
                if run == 3:
                    learner = SVC(kernel='poly', degree= 4, gamma = 'scale')
                    lname = 'SVM'
                    col = 'orange'
                if run == 4:
                    learner = MLPClassifier(hidden_layer_sizes=(100,50), solver='lbfgs', activation='relu')
                    lname = 'NueralNetwork'
                    col = 'black'

                start_time = time.time()
                mytraining_accuracy, mytest_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                stop_time = time.time()
                run_time = stop_time - start_time

                sys.stdout.write(Fore.BLACK + "\n" + name + ' ' + lname + "Run Time: " + Fore.RED + "{:.2} sec".format(run_time) + ' ')
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
                    learner = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth = 15, random_state=1)
                    lname = 'DecisionTree'
                    col = 'green'
                if run == 1:
                    learner = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=15, random_state=1), n_estimators=50)
                    lname = 'Boost'
                    col = 'blue'
                if run == 2:
                    learner = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='brute', metric='manhattan')
                    lname = 'KNN'
                    col = 'red'
                if run == 3:
                    learner = SVC(kernel='poly', degree= 4, gamma = 'scale')
                    lname = 'SVM'
                    col = 'orange'
                if run == 4:
                    learner = MLPClassifier(hidden_layer_sizes=(100,50), solver='lbfgs', activation='relu')
                    lname = 'NueralNetwork'
                    col = 'black'

                mytraining_accuracy, mytest_accuracy = self.crossValidation(all_x_data, all_y_data, learner)

                train_sizes, training_accuracy, test_accuracy = learning_curve(learner, all_x_data, all_y_data, train_sizes = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1], cv = 5)
                training_accuracy = training_accuracy.sum(axis = 1)/training_accuracy.shape[1]
                test_accuracy = test_accuracy.sum(axis = 1) / test_accuracy.shape[1]

                ax.plot(train_sizes, test_accuracy, label = lname + '_Test', linestyle='solid',linewidth=2, color = col)
                ax.plot(train_sizes, training_accuracy, label = lname +'Train',linestyle='dashed',linewidth=1, color = col)
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
                ns = [1,3,5,7,9,11,13,15,17,19,21,23,25]
                for i, p in enumerate(ps):
                    for j, n in enumerate(ns):
                        learner = DecisionTreeClassifier(criterion=p,min_samples_leaf=n, random_state=1)
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
                    learner = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth = n, random_state=1)
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
                ns = [10,20,30,40,50,60,70,80,90,100]
                for j, n in enumerate(ns):
                    learner = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',min_samples_leaf=12, random_state=1, max_depth=15), n_estimators=n)
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
                ns = [5,10,15,20,25,30]
                for j, n in enumerate(ns):
                    learner = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', min_samples_leaf=12, max_depth=n, random_state=1), n_estimators=50)
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
                ns = [10,20,30,40,50,60,70,80,90,100]
                for j, n in enumerate(ns):
                    learner = MLPClassifier(hidden_layer_sizes=(n, int(n/2)),solver='lbfgs', activation='relu', max_iter=500)
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n, training_accuracy, test_accuracy])
                ax.plot('n', 'Train_Accuracy', data=resultFrame, linewidth=1)
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
                ns = ['identity', 'tanh', 'relu','logistic']
                for j, n in enumerate(ns):
                    learner = MLPClassifier(hidden_layer_sizes=(100,50), solver='lbfgs', activation=n,max_iter=500)
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
                    learner = SVC(kernel= n, gamma = 'scale')
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


            #possible: C - penaluty parameter

            # Second Experiment
            if run == 1 or run == 3:
                ns = [1,2,3,4]
                for j, n in enumerate(ns):
                    learner = SVC(kernel='poly', degree= n, gamma = 'scale')
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

            #First Experiment
            if run == 0 or run == 2:
                ns = [5,10,15,20,25,30,35,40,45,50]
                for j,n in enumerate(ns):
                    learner = KNeighborsClassifier(n_neighbors=n, weights = 'uniform', algorithm = 'brute')
                    training_accuracy, test_accuracy = self.crossValidation(all_x_data, all_y_data, learner)
                    resultFrame.loc[len(resultFrame)] = list([n,training_accuracy,test_accuracy])
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
                plt.savefig('KNN_' + name + '_Results'+ str(run) + '.png')
                # plt.show()
                ax.clear()
                plt.clf()
                plt.cla()

            # Second Experiment
            if run == 1 or run == 3:
                
                ns = ['minkowski','euclidean','manhattan','chebyshev']
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

    def crossValidation(self, all_x_data, all_y_data, learner):
        # Run the selected model for each type of algorithm
        training_accuracy = []
        test_accuracy = []
        for i in range(5):
            xTrain, xTest, yTrain, yTest = train_test_split(all_x_data, all_y_data, test_size = 0.3, random_state = i)
            temp_train, temp_test = self.evaluateLeaner(xTrain, yTrain, xTest, yTest, learner)
            training_accuracy.append(temp_train)
            test_accuracy.append(temp_test)
        training_accuracy = sum(training_accuracy)/len(training_accuracy)
        test_accuracy = sum(test_accuracy)/len(test_accuracy)
        # sys.stdout.write(Fore.BLACK + "\nTrain Cross Validation Accuracy: " + Fore.BLUE + "{:.2%}".format(training_accuracy))
        # sys.stdout.write(Fore.BLACK + "\nTest Cross Validation Accuracy: " + Fore.BLUE + "{:.2%}".format(test_accuracy) + "\n")
        return training_accuracy, test_accuracy

    def evaluateLeaner(self, xTrain, yTrain, xTest, yTest, learner):
        #### Run the model on the Trainng data
        learner.fit(xTrain,np.ravel(yTrain))
        yPredictTraining = pd.DataFrame(data=learner.predict(xTrain), index = yTrain.index)
        training_accuracy = accuracy_score(yTrain, yPredictTraining)
        # sys.stdout.write(Fore.BLACK + "\nTraining Accuracy: " + Fore.RED + "{:.2%}".format(training_accuracy))

        #### Run the model on the Test data
        yPredictTest = pd.DataFrame(data=learner.predict(xTest), index=yTest.index)
        test_accuracy = accuracy_score(yTest, yPredictTest)
        # sys.stdout.write(Fore.BLACK + "\nTest Accuracy: " + Fore.RED + "{:.2%}".format(test_accuracy)+"\n")
        return training_accuracy, test_accuracy

    def getTechnicalIndicatorData(self):
        all_data = pd.read_csv('TechnicalIndicators.csv', parse_dates=True)
        all_data = all_data.drop('spy_Price', axis=1).drop('Daily_Return', axis=1)
        all_x_data = all_data.drop('Result', axis=1)
        all_y_data = pd.DataFrame(all_data['Result'], all_data.index)
        return all_x_data, all_y_data, "Technical Indicator Dataset"

    def getPokerData(self):
        all_data = pd.read_csv('poker.csv', parse_dates=True)
        all_x_data = all_data.drop('Class', axis=1)
        transformer = MinMaxScaler(feature_range=(0, 1)).fit(all_x_data)  # fit does nothing.
        all_x_data = transformer.transform(all_x_data)
        all_y_data = pd.DataFrame(all_data['Class'], all_data.index)
        return all_x_data, all_y_data, "Poker Dataset"
