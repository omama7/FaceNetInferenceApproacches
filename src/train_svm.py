from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import json
import pickle
import numpy as np
import time


def get_data():
    data_file = 'data/train.json'

    with open(data_file, 'r') as f:
        data = json.load(f)
    print('Loaded Data')
    X = []
    Y = []

    for key, values in data.items():
        for value in values:
            X.append(value)
            Y.append(int(key))
    return X, Y

def main():

    X, Y = get_data()


    param_grid = [
        {'C': [1, 10, 100, 1000],
         'kernel': ['linear']},
        # {'C': [1, 10, 100, 1000],
        #  'gamma': [0.001, 0.0001],
        #  'kernel': ['rbf']}
    ]
    print(len(X[0]))
    tic = time.time()

    clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)

    clf.fit(X, Y)
    toc = time.time()
    print('Training Time: ', toc - tic)
    predictions = clf.predict_proba([X[20]])[0]
    index = np.argmax(predictions)
    print(predictions[index], index)
    print(Y[20])

    with open('data/classifier.pkl', 'wb') as f:
        pickle.dump((clf), f)



if __name__ == '__main__':
    main()