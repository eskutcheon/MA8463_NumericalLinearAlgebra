import pandas as pd
import numpy as np
import seaborn as sbn
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from mcless import mcless
import feature_expansion as FE
import preprocessing_util as PREP

def get_test_accuracy(prediction, y_test, test_size):
    num_correct = np.sum([np.argmax(prediction[k]) == y_test[k] for k in range(test_size)])
    return num_correct/test_size

def test_feature_expansions(model, X_train, X_test, y_train, y_test):
    accuracy = np.zeros(5)
    test_size = len(y_test)
    mse_prediction = FE.test_mse_features(model, X_train, X_test, y_train, y_test)
    accuracy[0] = get_test_accuracy(mse_prediction, y_test, test_size)
    single_taylor_pred, multi_taylor_pred = FE.test_taylor_features(X_train, y_train, X_test)
    accuracy[1] = get_test_accuracy(single_taylor_pred, y_test, test_size)
    accuracy[2] = get_test_accuracy(multi_taylor_pred, y_test, test_size)
    mean_feat_pred = FE.test_mean_features(X_train, y_train, X_test)
    accuracy[3] = get_test_accuracy(mean_feat_pred, y_test, test_size)
    med_feat_pred = FE.test_med_features(X_train, y_train, X_test)
    accuracy[4] = get_test_accuracy(med_feat_pred, y_test, test_size)
    return accuracy

def get_acc_dict(num_runs):
    return {'MCLESS w/ original' : np.zeros(num_runs),
            'MCLESS w/ mse' : np.zeros(num_runs),
            'MCLESS w/ single taylor' : np.zeros(num_runs),
            'MCLESS w/ multi taylor' : np.zeros(num_runs),
            'MCLESS w/ col mean' : np.zeros(num_runs),
            'MCLESS w/ col median' : np.zeros(num_runs),
            'Naive Bayes w/ original' : np.zeros(num_runs),
            'K Neighbors w/ original' : np.zeros(num_runs),
            'Random Forest w/ original' : np.zeros(num_runs)}

def test_model_predictions(X_train, X_test, y_train, y_test):
    accuracy = np.zeros(9)
    model = mcless(X_train, y_train)
    model.compute_training_matrices()
    prediction = model.predict(X_test)
    accuracy[0] = get_test_accuracy(prediction, y_test, test_size)
    accuracy[1:6] = test_feature_expansions(model, X_train, X_test, y_train, y_test)
    accuracy[6:] = test_sklearn_models(X_train, X_test, y_train, y_test)
    return accuracy

def test_sklearn_models(X_train, X_test, y_train, y_test):
    names = ['Naive Bayes', 'K Neighbors', 'Random Forest']
    classifiers = [GaussianNB(),
                KNeighborsClassifier(7),
                RandomForestClassifier(max_depth=5, n_estimators=50, max_features=1)]
    accuracy = np.zeros(len(names))
    for clf, count in zip(names, classifiers, range(len(names))):
        clf.fit(X_train, y_train)
        accuracy[count] = clf.score(X_test, y_test)
    return accuracy

def print_mean_accuracy(acc_dict, header):
    print(header)
    for key, val in acc_dict.items():
        print(f'{key} : {np.mean(val)}')

def print_data_description(dataset, dataset_name, shape):
    print(f'MCLESS testing on sklearn {dataset_name} dataset')
    print('###########################################################################')
    print(dataset.DESCR, '\n')
    print('###########################################################################')
    print(f'{dataset_name} dataset features:\n{dataset.feature_names}\n')
    print(f'{dataset_name} dataset class labels:\n{dataset.target_names}\n')
    print(f'number of data samples: {shape[0]}\n')
    print(f'number of features per sample: {shape[1]}\n')

if __name__ == '__main__':
    datasets = [load_iris(), load_wine(), load_breast_cancer()]
    dataset_names = ['Iris', 'Wine', 'Breast Cancer'] # since not all have filenames apparently
    num_runs = 10
    num_datasets = len(datasets)
    set_count = 0
    for data_read in datasets:
        X = data_read.data
        y = data_read.target
        N, d = X.shape
        print_data_description(data_read, dataset_names[set_count], X.shape)
        label_set = set(y)
        num_classes = len(label_set)
        acc_dict = get_acc_dict(num_runs)
        acc_dict_norm = get_acc_dict(num_runs)
        for i in tqdm(range(num_runs), total=num_runs, desc=f'Progress on {dataset_names[set_count]} dataset'):
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=i,stratify=y)
            PREP.fill_nan(X_train)
            PREP.fill_nan(X_test)
            test_size = len(y_test)
            X_train_norm = PREP.normalize_data(X_train, d)
            X_test_norm = PREP.normalize_data(X_test, d)

            unnormed_acc = test_model_predictions(X_train, X_test, y_train, y_test)
            normed_acc = test_model_predictions(X_train_norm, X_test_norm, y_train, y_test)
            loop_count = 0
            for key in acc_dict.keys():
                acc_dict[key][i] = unnormed_acc[loop_count]
                acc_dict_norm[key][i] = normed_acc[loop_count]
                loop_count += 1
        unnormed_header = '\nmean accuracies for the test predictions of unnormed data:'
        normed_header = '\nmean accuracies for the test predictions of normed data:'
        print_mean_accuracy(acc_dict, unnormed_header)
        print_mean_accuracy(acc_dict_norm, normed_header)
        set_count += 1