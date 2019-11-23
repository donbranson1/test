How to run:
To run the code, you will need a version of python 3.7 and have at a minimum the following libraries installed:
unittest
pandas
numpy
sys
colorama
matplotlib
scikit-learn
time
scipy
mlrose
mdptoolbox

How to use:
The MachineLearning.py file contains a set of test cases needed to recreate the charts in the pdf analysis.  
To run each test, the user just needs to execute the test case and wait for the charts to be saved to the image folder. It's good practice to delete these files each round because they will be overwritter otherwise. All charts are documented under one of the test cases.  What is not documented are the numeruous initial runs needed to arrive at the default hyperparameters before these tests begin. I performed this analysis using pycharm which has support for running unit tests.

Where to get the data and code:
Based on Piazza comments my code and data are linked here: https://github.gatech.edu/dbranson3/CS7641_PUBLIC/tree/master/Assignment3
For redundancy purposes the files are also located here with additional documentation: https://github.com/donbranson1/test/tree/master

Finally, I had to perform many modifications to mdptoolbox to ensure accuracy and alignment with lecture notes. This code is documented in the github repos and it should be called versus importing mdptoolbox directly.
