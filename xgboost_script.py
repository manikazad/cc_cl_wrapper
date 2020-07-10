

import numpy as np
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import pandas as pd
import pickle
import sys
from pprint import pprint


def main():

    parser = argparse.ArgumentParser()

    # Method Argument
    parser.add_argument("-V", "--verbose", help="Enable verbose output.", action='store_true', default=False)

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
    parser.add_argument("--feature_imp", action='store_true', help="Return feature importance")
    parser.add_argument("--result_path", action='store', help="Result output file path.")
    parser.add_argument("--input_model", action='store', help="Input Model Address")
    parser.add_argument("--model_output_path", action='store', help="Output Model File Path")


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
    parser.add_argument("--error_metric", action='store', default='mse', help="Error metric to be used. Available options are "
                                                               "'mse'(mean squared error), 'mae' (mean absolute error) and 'rmse'")
    parser.add_argument("-l", "--loss",
                        help="Loss function to be optimized. Default is 'ls' least square loss, Other options are: \
                        'lad': Least Absolute Devaition",
                        default='ls', action='store')
    parser.add_argument("-r", "--learning_rate", type=float,
                        help="Learning rate: learning rate shrinks the contribution of each tree by `lrate`.",
                        default=0.1, action='store')
    parser.add_argument("-n", "--n_estimators", type=int, help="The number of boosting stages to perform", default=100)
    parser.add_argument("--subsample", type=float,
                        help="The fraction of samples to be used for fitting the individual base learners.",
                        default=1.0, action='store')
    parser.add_argument("-d", "--max_depth", type=int, help="Maximum depth upto which the trees should grow.",
                        default=3, action='store')
    parser.add_argument("--max_features", type=int, help="Number of features to be included in the final model",
                        action='store', default=None)
    parser.add_argument("--min_samples_split", type = int,
                        help = "The minimum number of samples required to split an internal node", action = 'store', default=2)
    parser.add_argument("--min_samples_leaf", type=int,
                        help="The minimum number of samples required to be at a leaf node", action='store', default=1)

    parsed_args = parser.parse_args()

    # Checking for necessary input arguments in each case

    if parsed_args.run_test:
        X = load_data('./data/X_train_test.csv')
        Y = load_data('./data/Y_train_test.csv')
        X, Y = shuffle(X, Y, random_state=13)
        print("Data Loaded with {} rows and {} columns".format(X.shape[0], X.shape[1]), file=sys.stdout)
        assert X.shape[0] == 506, "Test failed, row counts didn't match"
        assert X.shape[1] == 13, "Test failed, column counts didn't match"

        X_train, Y_train, X_test, Y_test = train_test_split(X, Y)
        print("Data split into train ({} rows) and test ({} rows)".format(X_train.shape[0], X_test.shape[0]), file=sys.stdout)
        assert X_train.shape[0] == 404, "Test failed, train rows counts didn't match"
        assert X_test.shape[0] == 102, "Test failed, test rows counts didn't match"

        params = get_params(parsed_args)
        print("Model Parameters:")
        pprint(params)

        assert params == {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'loss': 'ls', 'subsample': 1.0,
                          'min_samples_split': 2, 'min_samples_leaf': 1, 'verbose': False}, "Test failed: Parameter values didn't match"

        model = train_model(X_train, Y_train, params)
        print("Model Trained", file=sys.stdout)
        test_score = test_model(model, X_test, Y_test)
        print("Model Tested over test data, Test score ('mse') is :", test_score)
        feat_importance = get_feature_importance(model, X.columns)
        print(feat_importance)

    if parsed_args.predict:
        assert parsed_args.input_model, "Please provide an Input Model Path"
        assert parsed_args.X_predict, "Please provid X Variables for prediction"
        assert parsed_args.result_file, "Please provide output result file path"

        X_predict = load_data(parsed_args.X_predict)
        model = load_model(parsed_args.input_model)
        predicted = model.predict(X_predict)
        print("Prediction Successful.")
        print("The Results are being dumped at following location: ", parsed_args.result_path)
        with open(parsed_args.result_path, 'w') as _fp:
            _fp.write(predicted)

    elif parsed_args.train_test:
        if parsed_args.pre_split:
            assert parsed_args.X_train, "Please provide Training Data X"
            assert parsed_args.Y_train, "Please provide Training Data Y"
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
            assert parsed_args.model_output_path, "Please provide a model output path"
            # assert parsed_args.train_test_split, "Please provide split ratio between train and test data"

            X = load_data(parsed_args.X)
            Y = load_data(parsed_args.Y)
            X_train, Y_train, X_test, Y_test = train_test_split(X, Y, parsed_args.train_test_split)

        params = get_params(parsed_args)
        model = train_model(X_train, Y_train, params)
        dump_model(model, parsed_args.model_output_path)
        print("The model is dumped at following location:", parsed_args.model_output_path)

        test_score = test_model(model, X_test, Y_test, error_metric=parsed_args.error_metric)
        print("Test Score ({}): ".format(parsed_args.error_metric), test_score)

        if parsed_args.feature_imp:
            feat_importance = get_feature_importance(model, X.columns)
            print("Feature Importance: \n", feat_importance)


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

        if parsed_args.feature_imp:
            feat_importance = get_feature_importance(model, X.columns)
            print("Feature Importance: \n", feat_importance)

    elif parsed_args.test:
        assert parsed_args.X_test, "Please provide Testing data X"
        assert parsed_args.Y_test, "Please provide Testing data Y"
        assert parsed_args.input_model, "Please provide Input model path"

        model = load_model(parsed_args.input_model)
        X_test = load_data(parsed_args.X_test)
        Y_test = load_data(parsed_args.Y_test)
        test_score = test_model(model, X_test, Y_test, error_metric=parsed_args.error_metric)
        print("Test Score ({}): ".format(parsed_args.error_metric), test_score)


def train_test_split(X, Y, split_perc=0.8,do_shuffle=True):
    if do_shuffle:
        X, Y = shuffle(X, Y, random_state=13)
    offset = int(X.shape[0] * split_perc)

    X_train, Y_train = X[:offset], Y[:offset]
    X_test, Y_test = X[offset:], Y[offset:]

    return X_train, Y_train, X_test, Y_test


def get_feature_importance(model, feature_names):
    feature_importance = model.feature_importances_

    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)

    importance_df = pd.DataFrame([feature_names, feature_importance[sorted_idx]]).T
    importance_df.columns = ['feature', 'importance']

    feat_imp = importance_df.sort_values('importance', ascending=False)
    feat_imp.to_csv("./feature_importance.csv")

    return feat_imp


def load_model(model_path):
    return pickle.load(model_path)


def dump_model(model, output_path):
    with open(output_path, 'wb') as _fp:
        pickle.dump(model, _fp, pickle.HIGHEST_PROTOCOL)


def load_data(data_path):
    return pd.read_csv(data_path)


def train_model(X_train, Y_train, params):
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, Y_train)
    return clf


def test_model(model, X_test, y_test, error_metric='mse'):
    if error_metric == 'mse':
        err = mean_squared_error(y_test, model.predict(X_test))
    elif error_metric == 'mae':
        err = mean_absolute_error(y_test, model.predict(X_test))
    elif error_metric == 'rmse':
        err = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return err


def get_params(parsed_args):
    params = {'n_estimators': parsed_args.n_estimators,
              'max_depth': parsed_args.max_depth,
              'learning_rate': parsed_args.learning_rate,
              'loss': parsed_args.loss,
              'subsample': parsed_args.subsample,
              "min_samples_split": parsed_args.min_samples_split,
              'min_samples_leaf': parsed_args.min_samples_leaf,
              'verbose': parsed_args.verbose,
              }
    return params


if __name__ == "__main__":
    main()
