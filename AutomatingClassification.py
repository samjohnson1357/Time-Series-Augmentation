"""
File Summary: This file seeks to automate the tasks of cross validation, hyperparameter tuning, and accuracy evaluation.
"""

# Data preprocessing
import pandas as pd
import numpy as np

# Modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Scoring and plotting
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class AutomaticClassification:

  def __init__(self, trainX, trainY, testX, testY, nFolds=5):
    self.trainX = trainX
    self.trainY = trainY
    self.testX = testX
    self.testY = testY
    self.nFolds = nFolds

  def runCrossValidation(self, model):
    scores = cross_val_score(model, self.trainX, self.trainY, cv=self.nFolds)
    return scores.mean()

  def createModel(self, modelClass, **kwargs):
    """
    Helper function to create the model for the runCrossValidationAcrossHyperparameters method.
    The parameters of the model (including the hyperparameter, should be passed in here).
    """
    return modelClass(**kwargs)

  def runCrossValidationAcrossHyperparameters(self, modelClass, hyperparamStr, hyperparamValList, modelStr, xAxis, **kwargs):
      """
      Function runs cross validation across a list of hyperparameters for a given model.
      The regular parameters for the model class should be given in kwargs.
      The hyperparameter values will be given in hyperparamValList.
      """
      accuracyList = []
      for val in hyperparamValList:
          
          # Add the hyperparameter value to the keyword dictionary.
          kwargs[hyperparamStr] = val
          
          model = self.createModel(modelClass, **kwargs)
          
          accuracy = self.runCrossValidation(model)
          accuracyList.append(accuracy)
      
      plt.figure( figsize=(20, 10) )
      plt.plot(hyperparamValList, accuracyList, color='red')
      plt.xlabel(xAxis, fontsize=15)
      plt.ylabel('Accuracy', fontsize=15)
      plt.title('{} {}-Fold Cross Validated Accuracy by Hyperparameter {}'.format(modelStr, self.nFolds, xAxis), fontsize=20)
      plt.savefig(modelStr + ' Cross Validation Plot.png', bbox_inches='tight')
      
      return accuracyList

  def getBestHyperparameter(self, hyperparamValList, accuracyList):
    """
    Function returns the best hyperparameter value and top accuracy from CV
    as (hyperparameter, accuracy).
    """
    bestHyper = hyperparamValList[np.argmax(accuracyList)]
    bestAccuracy = np.max(accuracyList)
    return bestHyper, bestAccuracy

  def getAccuracyOnTestSet(self, model):
      model.fit(self.trainX, self.trainY)
      return model.score(self.testX, self.testY)

  def logisticRegression(self):
    print('Now fitting logistic regression')
    hyperparamValList = np.linspace(0.01, 2, 10)
    accList = self.runCrossValidationAcrossHyperparameters(LogisticRegression, 'C', hyperparamValList, 'Logistic Regression', 
                                                       'C', penalty='l2', max_iter=100, n_jobs=-1)
    bestCVHyper, bestCVAcc = self.getBestHyperparameter(hyperparamValList, accList)
    model = LogisticRegression(penalty='l2', C=bestCVHyper)
    testAcc = self.getAccuracyOnTestSet(model)
    return ['Logistic Regression', 'C', bestCVHyper, bestCVAcc, testAcc]

  def decisionTree(self):
    print('Now fitting decision tree')
    hyperparamValList = list(range(1, 40, 5))
    accList = self.runCrossValidationAcrossHyperparameters(DecisionTreeClassifier, 'max_depth', hyperparamValList, 
                                                       'Decision Tree', 'Max Depth')
    bestCVHyper, bestCVAcc = self.getBestHyperparameter(hyperparamValList, accList)
    model = DecisionTreeClassifier(max_depth=bestCVHyper)
    testAcc = self.getAccuracyOnTestSet(model)
    return ['Decision Tree', 'Max depth', bestCVHyper, bestCVAcc, testAcc]

  def kNearestNeighbors(self, nNeighbors=None):
    """
    If the hyperparameter nNeighbors is set to a value, then hyperparameter tuning through CV will not be run.
    """
    print('Now fitting k-nearest-neighbors')

    # Tune hyperparameters if desired
    if nNeighbors is None:
        hyperparamValList = list(range(1, 100, 10))
        accList = self.runCrossValidationAcrossHyperparameters(KNeighborsClassifier, 'n_neighbors',
                                                          hyperparamValList, 'K Nearest Neighbors', 'N-Neighbors')
        nNeighbors, bestCVAcc = self.getBestHyperparameter(hyperparamValList, accList)
    else:
        bestCVAcc = '-'

    # Calculate accuracy on the test set
    model = KNeighborsClassifier(n_neighbors=nNeighbors)
    testAcc = self.getAccuracyOnTestSet(model)
    return ['KNN', 'n_neighbors', nNeighbors, bestCVAcc, testAcc]


  def supportVectorMachine(self):
    print('Now fitting support vector machine')
    hyperparamValList = np.linspace(0.01, 10, 10)
    accList = self.runCrossValidationAcrossHyperparameters(LinearSVC, 'C', 
                                                      hyperparamValList, 'Support Vector Machine', 'C')
    bestCVHyper, bestCVAcc = self.getBestHyperparameter(hyperparamValList, accList)
    model = LinearSVC(C=bestCVHyper)
    testAcc = self.getAccuracyOnTestSet(model)
    return ['Support Vector Machine', 'C', bestCVHyper, bestCVAcc, testAcc]

  def randomForest(self):
    """
    In general, random forests don't overfit to the training data as more 
    trees are added. Thus, cross validation is unnecessary.
    """
    print('Now fitting random forest classifier')
    model = RandomForestClassifier(n_estimators=300)
    testAcc = self.getAccuracyOnTestSet(model)
    return ['Random Forest', 'n_estimators', '-', '-', testAcc]

  def gradientBoosting(self):
    print('Now fitting gradient boosting classifier')
    hyperparamValList = list(range(50, 510, 100))
    accList = self.runCrossValidationAcrossHyperparameters(GradientBoostingClassifier, 'n_estimators', hyperparamValList, 
                                                        'Gradient Boosting Machine', 'N-Estimators')
    bestCVHyper, bestCVAcc = self.getBestHyperparameter(hyperparamValList, accList)
    model = GradientBoostingClassifier(n_estimators=bestCVHyper)
    testAcc = self.getAccuracyOnTestSet(model)
    return ['Gradient Boosting', 'Max depth', bestCVHyper, bestCVAcc, testAcc]


  def main(self, classifierArray=(True, True, True, True, True, True)):
    """
    The function runs cross validation, chooses the best hyperparameter,
    and then evaluates the model on a test set. It reports the best
    cross validated accuracy and the testing accuracy.

    Use the 'classifierArray' parameter to decide which models to run.
    It is simply a boolean array containing either True to run the model
    or False to not run the model.
    """

    # Fit the desired models.
    methodArray = np.array([self.logisticRegression, self.decisionTree, self.kNearestNeighbors, 
                            self.supportVectorMachine, self.randomForest, self.gradientBoosting])
    allRows = []
    for method in methodArray[classifierArray]:
      row = method()
      allRows.append(row)

    # Create a dataframe of the results.
    columns = ['Model', 'Hyperparam', 'Hyperparam val', 'CV Accuracy', 'Test Accuracy']
    reportDf = pd.DataFrame(data=allRows, columns=columns)
    reportDf.set_index('Model', inplace=True)

    return reportDf

