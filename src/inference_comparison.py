import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import json
import pickle
import numpy as np
import time
from create_comparison_dataset import get_data

def svm_inference():
    
    with open('data/classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open('data/label.json', 'r') as f:
        labels = json.load(f)
        labels = {v:k for k,v in labels.items()}

    BASE_DIR = 'data/images/inference/'

    for person in os.listdir(BASE_DIR):
        
        DIR = BASE_DIR + person + '/'
        emb_data = get_data(DIR)

        tic = time.time()
        predictions = clf.predict_proba(emb_data)
        toc = time.time()

        print('Time Takes For 1 Prediction: ', (toc - tic) / len(emb_data) )

        print('\n\nRunning SVM Inference For {}:'.format(person))
        for i in predictions:
            # print('\nAll Prediction: ', i)

            print(labels[str(np.argmax(i))], max(i))


def knn_inference():
    with open('data/train.json', 'r') as f:
        stored_data = json.load(f)

    with open('data/label.json', 'r') as f:
        labels = json.load(f)
        labels = {v:k for k,v in labels.items()}


    BASE_DIR = 'data/images/inference/'
    for person in os.listdir(BASE_DIR):
        DIR = BASE_DIR + person + '/'
        emb_data = get_data(DIR)

        print('\n\nRunning KNN Inference For {}:\n'.format(person))
        for i in emb_data:
            tic = time.time()
            for k,v in stored_data.items():
                count = 0
                for feat in v:
                    np_array1 = np.array(feat)
                    np_array2 = np.array(i)
                    dist = np.sqrt(np.sum(np.square(np.subtract(np_array1, np_array2))))
                    if dist < 0.8:
                        count += 1

                # print('Matches with {}: {} out of {}'.format(labels[k] , count, len(v)))
            toc= time.time()
            print('Time taken for 1 KNN inference: ', toc - tic)




def distance_inference():
    with open('data/train.json', 'r') as f:
        stored_data = json.load(f)

    with open('data/label.json', 'r') as f:
        labels = json.load(f)
        labels = {v:k for k,v in labels.items()}


    BASE_DIR = 'data/images/inference/'
    for person in os.listdir(BASE_DIR):
        DIR = BASE_DIR + person + '/'
        emb_data = get_data(DIR)

        print('\n\nRunning Distance Inference For {}:\n'.format(person))
        for i in emb_data:
            min_dis = 9999
            min_arg = 0
            for k,v in stored_data.items():
                np_array1 = np.zeros(512)
                for feat in v:
                    np_array1 = np_array1 + np.array(feat)
                
                np_array1 = np_array1 / len(v)

                np_array2 = np.array(i)

                dist = np.sqrt(np.sum(np.square(np.subtract(np_array1, np_array2))))

                if dist < min_dis:
                    min_dis = dist
                    min_arg = k

            print('Distance with {} is {}'.format(labels[min_arg] , min_dis))



if __name__ == '__main__':
    distance_inference()
    knn_inference()
    svm_inference()
