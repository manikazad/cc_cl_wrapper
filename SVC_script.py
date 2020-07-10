

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score
import argparse
import pandas as pd
import pickle
import sys
from pprint import pprint

def main():
    parser = argparse.ArgumentParser()

    # Method Argument
    parser.add_argument("--verbose", action='store_true', help="verbose : bool, optional, default False", default=False)

    parser.add_argument("--run_test", action='store_true', default=False, help="Runs test")

    parser.add_argument("--train", action='store_true', help='Boolean argument for running the training over data',
                        default=False)
    parser.add_argument("--test", action='store_true', help='Boolean argument for running the testing over pretrained ' \
                                                            'model. Input model file is required', default=False)
    parser.add_argument("-T", "--train_test", action='store_true', help='Boolean argument for running training and '
                                                                        'testing simultaneously', default=False)
    parser.add_argument("-P", "--predict", action='store_true',
                        help='Boolean argument for running the predictions on a '
                             'pretrained model. Model input file is neccessary, Returns predicted values. and stores '
                             'them in result file', default=False)
    parser.add_argument("--result_path",
                        action='store',
                        help="Result output file path.",
                        default=sys.stdout
                        )
    parser.add_argument("--input_model",
                        action='store',
                        help="Input Model Address",
                        )
    parser.add_argument("--model_output_path",
                        action='store',
                        help="Output Model File Path",
                        )

    # Data Arguments

    parser.add_argument("-X", action='store', help="Train Test X Data path(csv only, if the data is not splitted beforehand")
    parser.add_argument("-Y", action='store', help="Train Test Y Data path(csv only), if the data is not splitted beforehand")
    parser.add_argument("-X_train", help="Regressor(X) data path (csv only)", action='store')
    parser.add_argument("-Y_train", help="Label(Y) data path (csv only)",  action='store')
    parser.add_argument("-X_test", help="Regressor(X) data path (csv only)",  action='store')
    parser.add_argument("-Y_test", help="Label(Y) data path (csv only)", action='store')
    parser.add_argument("-X_predict", help="Regressor(X) data path (csv only)", action='store')

    parser.add_argument("-S", "--train_test_split", type=float, help="Fraction of data to be used for training the model",
                        default=0.8, action='store')
    parser.add_argument("--pre_split", help="Boolean Indicator variable, which tells if the train test data is splitted beforehand",
                        action='store_true', default=False)

    # Model Parameter Arguments

    parser.add_argument("--error_metric",
                        help="Error metric to be used. Available options are \n"
                             "'clf_report'(Classfication Report: Build a text report showing the main classification metrics ), \n"
                             "'conf_mat' (Confusion Matrix: Compute confusion matrix to evaluate the accuracy of a classification), \n"
                             " 'f1_score' (Compute the F1 score, also known as balanced F-score or F-measure) \n"
                             "'prfs' (Precision Recall FScore and Support for each class) \n"
                             "'precision' (Compute the precision) \n"
                             "'recall' (Compute the recall) ",
                        default='clf_report',
                        action='store',)
    parser.add_argument("-C","--C",
                        dest="C",
                        help="float, optional (default=1.0) \n"
                             "Penalty parameter C of the error term.",
                        type=float,
                        default=1.0,
                        action='store',)
    parser.add_argument("-k", "--kernel",
                        dest='kernel',
                        help="string, optional (default='rbf') \n"
                             "Specifies the kernel type to be used in the algorithm. \n"
                             "It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.",
                        default='rbf',
                        action='store')
    parser.add_argument("-d", "--degree",
                        dest='degree',
                        help="int, optional (default=3)\n"
                                "Degree of the polynomial kernel function ('poly').\n"
                                "Ignored by all other kernels.",
                        type=int,
                        default= 3,
                        action='store')
    parser.add_argument("--gamma",
                        help="float, optional (default='auto')\n"
                             "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.\n"
                             "If gamma is 'auto' then 1/n_features will be used instead.",
                        default='auto',
                        action = 'store')
    parser.add_argument("--coef0",
                        help="float, optional (default=0.0) \n"
                             "Independent term in kernel function.\n"
                             "It is only significant in 'poly' and 'sigmoid'.",
                        type = float,
                        default=0.0,
                        action='store')
    parser.add_argument("--probability",
                        help="boolean, optional (default=False) \n"
                             "Whether to enable probability estimates. This must be enabled prior \n"
                             "to calling `fit`, and will slow down that method.",
                        default=False,
                        action='store_true')
    parser.add_argument("--no_shrinking",
                        dest='shrinking',
                        help="boolean, optional (default=True) \n"
                             "Whether to use the shrinking heuristic.",
                        default=True,
                        action='store_false',
                        )
    parser.add_argument("--tol",
                        help="float, optional (default=1e-3) \n"
                             "Tolerance for stopping criterion.",
                        type=float,
                        default=1e-3,
                        action='store',
                        )
    parser.add_argument("--max_iter",
                        help="int, optional (default=-1) \n"
                             "Hard limit on iterations within solver, or -1 for no limit.",
                        type=int,
                        default=-1,
                        action='store'
                        )

    parsed_args = parser.parse_args()

    # Checking for necessary input arguments in each case

    if parsed_args.run_test:

        X = load_data('./data/X_train_test_class.csv')
        Y = load_data('./data/Y_train_test_class.csv')
        X, Y = shuffle(X, Y, random_state=13)
        print("Data Loaded with {} rows and {} columns".format(X.shape[0], X.shape[1]), file=sys.stdout)
        assert X.shape[0] == 150, "Test failed, row counts didn't match"
        assert X.shape[1] == 4, "Test failed, column counts didn't match"

        X_train, Y_train, X_test, Y_test = train_test_split(X, Y)
        print("Data split into train ({} rows) and test ({} rows)".format(X_train.shape[0], X_test.shape[0]), file=sys.stdout)
        assert X_train.shape[0] == 120, "Test failed, train rows counts didn't match"
        assert X_test.shape[0] == 30, "Test failed, test rows counts didn't match"

        params = get_params(parsed_args)
        print("Model Parameters:")
        pprint(params)

        # assert params == , "Test failed: Parameter values didn't match"

        model = train_model(X_train, Y_train, params)
        print("Model Trained", file=sys.stdout)

        print(X_test, Y_test)
        test_score = test_model(model, X_test, Y_test)
        print("Model Tested over test data, Test score (Classification report) is :\n", test_score)

    if parsed_args.predict:
        assert parsed_args.input_model, "Please provide an Input Model Path"
        assert parsed_args.X_predict, "Please provide X Variables for prediction"
        assert parsed_args.result_path, "Please provide output result file path"

        X_predict = load_data(parsed_args.X_predict)
        model = load_model(parsed_args.input_model)

        predicted = model.predict(X_predict)
        print("Prediction Successful.")
        print("The Results are being dumped at following location: ", parsed_args.result_path)
        with open(parsed_args.result_path, 'w') as _fp:
            _fp.write(predicted)

    elif parsed_args.train_test:
        if parsed_args.pre_split:
            assert parsed_args.X_train, "Please provide an Training Data X"
            assert parsed_args.Y_train, "Please provide an Training Data Y"
            assert parsed_args.X_test, "Please provide an Testing data X"
            assert parsed_args.Y_test, "Please provide an Testing data Y"
            assert parsed_args.model_output_path, "Please provide a model output path"

            X_train = load_data(parsed_args.X_train)
            Y_train = load_data(parsed_args.Y_train)
            X_test = load_data(parsed_args.X_test)
            Y_test = load_data(parsed_args.Y_test)

        else:
            assert parsed_args.X, "Please provide X data"
            assert parsed_args.Y, "Please provide Y data"
            assert parsed_args.train_test_split, "Please provide split ratio between train and test data"

            X = load_data(parsed_args.X)
            Y = load_data(parsed_args.Y)
            X_train, Y_train, X_test, Y_test = train_test_split(X, Y, parsed_args.train_test_split)

            assert X_train.shape[0] == 120
            assert X_test.shape[0] == 30
            assert Y_test.shape[0] == 30
            assert Y_train.shape[0] == 120

        params = get_params(parsed_args)
        print("Model Parameters: ")
        pprint(params)
        model = train_model(X_train, Y_train, params)
        dump_model(model, parsed_args.model_output_path)
        print("The model is dumped at following location:", parsed_args.model_output_path)

        print(X_test, Y_test)
        test_score = test_model(model, X_test, Y_test, error_metric=parsed_args.error_metric)
        print("Test Score ({}): \n".format(parsed_args.error_metric), test_score)


    elif parsed_args.train:
        assert parsed_args.X_train, "Please provide Training data X"
        assert parsed_args.Y_train, "Please provide Training data Y"
        assert parsed_args.model_output_path, "Please provide a model output path"

        X_train = load_data(parsed_args.X_train)
        Y_train = load_data(parsed_args.Y_train)

        params = get_params(parsed_args)
        model = train_model(X_train, Y_train, params)
        dump_model(model, parsed_args.model_output_path)
        print("The trained model has been dumped at following location ", parsed_args.model_output_path)

    elif parsed_args.test:
        assert parsed_args.X_test, "Please provide Testing data X"
        assert parsed_args.Y_test, "Please provide Testing data Y"
        assert parsed_args.input_model, "Please provide Input model path"

        model = load_model(parsed_args.input_model)
        X_test = load_data(parsed_args.X_test)
        Y_test = load_data(parsed_args.Y_test)
        test_score = test_model(model, X_test, Y_test, error_metric=parsed_args.error_metric)
        print("Test Score ({}): ".format(parsed_args.error_metric), test_score)


def train_test_split(X, Y, split_perc=0.8, do_shuffle=True):
    if do_shuffle:
        X, Y = shuffle(X, Y, random_state=13)
    offset = int(X.shape[0] * split_perc)

    X_train, Y_train = X[:offset], Y[:offset]
    X_test, Y_test = X[offset:], Y[offset:]

    return X_train, Y_train, X_test, Y_test


def load_model(model_path):
    return pickle.load(model_path)


def dump_model(model, output_path):
    with open(output_path, 'wb') as _fp:
        pickle.dump(model, _fp, pickle.HIGHEST_PROTOCOL)


def load_data(data_path):
    return pd.read_csv(data_path)


def test_model(model, X_test, y_test, error_metric='clf_report'):
    if error_metric == 'clf_report':
        err = classification_report(y_test, model.predict(X_test))
    elif error_metric == 'conf_mat':
        err = confusion_matrix(y_test, model.predict(X_test))
    elif error_metric == 'prfs':
        err = precision_recall_fscore_support(y_test, model.predict(X_test))
    elif error_metric == 'precision':
        err = precision_score(y_test, model.predict(X_test))
    elif error_metric == 'recall':
        err = recall_score(y_test, model.predict(X_test))
    return err


def train_model(X_train, Y_train, params):
    if params['kernel'] == 'linear':
        svclassifier = LinearSVC(**params)
        svclassifier.fit(X_train, Y_train)
    else:
        svclassifier = SVC(**params)
        svclassifier.fit(X_train, Y_train)
    return svclassifier


def get_params(parsed_args):
    params = {
            'C': parsed_args.C,
            'kernel': parsed_args.kernel,
            'degree': parsed_args.degree,
            'gamma': parsed_args.gamma,
            'coef0': parsed_args.coef0,
            'shrinking': parsed_args.shrinking,
            'probability': parsed_args.probability,
            'max_iter': parsed_args.max_iter,
            'tol': parsed_args.tol,
            'verbose': False
            }

    return params


if __name__ == "__main__":
    main()
