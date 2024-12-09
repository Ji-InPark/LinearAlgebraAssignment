import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class class_digit_classification():
    def __init__(self):
        self.testY = None
        self.testX = None

        self.trainX = None
        self.trainY = None

        self.predictY = None

        self.load_data()

    def load_data(self):
        train_data = pd.read_csv('train.csv')

        self.trainX = train_data.drop('label', axis=1)
        self.trainX = np.array(self.trainX)

        self.trainY = train_data['label']
        self.trainY = np.array(self.trainY)

        test_data = pd.read_csv('test.csv')

        self.testX = test_data
        self.testX = np.array(self.testX)

    def train_and_test_svm_model(self):
        model = svm.SVC()
        model.fit(self.trainX, self.trainY)
        self.predictY = model.predict(self.testX)

    def train_and_test_logistic_regression_model(self):
        model = LogisticRegression()
        model.fit(self.trainX, self.trainY)
        self.predictY = model.predict(self.testX)

    def train_and_test_random_forest_model(self):
        model = RandomForestClassifier(n_estimators=5, random_state=0)
        model.fit(self.trainX, self.trainY)
        self.predictY = model.predict(self.testX)

    def write_results(self, file_name):
        ids = [i for i in range(1, len(self.predictY) + 1)]
        results = pd.DataFrame({'ImageId': ids, 'Label': self.predictY})
        results.to_csv(f'{file_name}.csv', index=False)


if __name__ == '__main__':
    clf = class_digit_classification()

    clf.train_and_test_svm_model()
    clf.write_results('svm_results')

    clf.train_and_test_logistic_regression_model()
    clf.write_results('logistic_regression_results')

    clf.train_and_test_random_forest_model()
    clf.write_results('random_forest_results')
