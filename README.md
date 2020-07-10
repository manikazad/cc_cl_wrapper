# cc_cl_wrapper

## Neural Network -- Command Line Usage

	  -h, --help            	show this help message and exit
	  -V, --verbose         	Flag argument, Enable verbose output.
	  --run_test           	 Flag argument which runs the tests
	  --train               	Boolean flag argument for running the training over data
	  --test                	Boolean flag argument for running the testing over pretrained model. Input model file is required
	  -T, --train_test      	Boolean flag argument for running training and testing simultaneously
	  -P, --predict         	Boolean flag argument for running the predictions on a pretrained model. Model input file is neccessary, Returns predicted values. and stores them in result file
	  --feature_imp         	 string, *.csv file, deafult None, Return feature importance
	  --result_path RESULT_PATH	string,*.csv file, deafult None, Result output file path.
	  --input_model INPUT_MODEL	string, deafult None, Input Model Address
	  --model_output_path MODEL_OUTPUT_PATH	 string, deafult None, Output Model File Path
	  -X X                  	string, *.csv file , default None, Train Test X Data path(csv only, if the data is not splitted beforehand
	  -Y Y                  	string, *.csv file , default None, Train Test Y Data path(csv only), if the data is not  splitted beforehand
	  -X_train X_TRAIN      	string, *.csv file , default None, Regressor(X) data path (csv only)
	  -Y_train Y_TRAIN      	string, *.csv file , default None, Label(Y) data path (csv only)
	  -X_test X_TEST        	string, *.csv file , default None , Regressor(X) data path (csv only)
	  -Y_test Y_TEST        	string, *.csv file , default None, Label(Y) data path (csv only)
	  -X_predict X_PREDICT  	string, *.csv file , default None Regressor(X) data path (csv only)
	  -S TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT	 float, default 0.9, Fraction of data to be used for training the model
	  --pre_split           	Bool, default false, Indicator variable, which tells if the train test data is splitted beforehand
	  --error_metric ERROR_METRIC	string, default 'mse', Error metric to be used. options are  'mse'(mean squared error), 'mae' (mean absolute error) and 'rmse'
	  -l LOSS, --loss LOSS  	Loss function to be optimized. Default is 'ls' least square loss, Other options are: 'lad': Least Absolute Devaition
		
		
	 --hidden_layer_sizes	 element represents the number of neurons in the ith hidden layer.
	  -a ACTIVATION, --activation ACTIVATION	options: {'identity', 'logistic', 'tanh', 'relu'}, default 'relu', Activation function for the hidden layer.
	  -r SOLVER, --solver SOLVER	                        options: {'lbfgs', 'sgd', 'adam'}, default 'adam'
	  --alpha ALPHA         	 float, optional, default 0.0001, L2 penalty  (regularization term) parameter
	  --batch_size BATCH_SIZE	int, optional, default 'auto', Size of minibatches for stochastic optimizers If the solver is 'lbfgs', the classifier will not use minibatch. When set to 'auto',classifier will not use minibatch.  batch_size=min(200, n_samples) 
	  --learning_rate LEARNING_RATE	 {'constant', 'invscaling', 'adaptive'}, default  'constant' Learning rate schedule for weight updates.
	  --max_iter MAX_ITER   	int, optional, default 200 Maximum number of  iterations. The solver iterates until convergence(determined by 'tol') or this number of iterations.
	  --tol TOL            	 toleralnce : float, optional, default 1e-4
	  --early_stopping      	bool, default False

## XGB -- Command Line Usage

	  -h, --help            	show this help message and exit
	  -V, --verbose         	Flag argument, Enable verbose output.
	  --run_test           	 Flag argument which runs the tests
	  --train               	Boolean flag argument for running the training over data
	  --test                	Boolean flag argument for running the testing over pretrained model. Input model file is required
	  -T, --train_test      	Boolean flag argument for running training and testing simultaneously
	  -P, --predict         	Boolean flag argument for running the predictions on a pretrained model. Model input file is neccessary, Returns predicted values. and stores them in result file
	  --feature_imp         	 string, *.csv file, deafult None, Return feature importance
	  --result_path RESULT_PATH	string,*.csv file, deafult None, Result output file path.
	  --input_model INPUT_MODEL	string, deafult None, Input Model Address
	  --model_output_path MODEL_OUTPUT_PATH	 string, deafult None, Output Model File Path
	  -X X                  	string, *.csv file , default None, Train Test X Data path(csv only, if the data is not splitted beforehand
	  -Y Y                  	string, *.csv file , default None, Train Test Y Data path(csv only), if the data is not  splitted beforehand
	  -X_train X_TRAIN      	string, *.csv file , default None, Regressor(X) data path (csv only)
	  -Y_train Y_TRAIN      	string, *.csv file , default None, Label(Y) data path (csv only)
	  -X_test X_TEST        	string, *.csv file , default None , Regressor(X) data path (csv only)
	  -Y_test Y_TEST        	string, *.csv file , default None, Label(Y) data path (csv only)
	  -X_predict X_PREDICT  	string, *.csv file , default None Regressor(X) data path (csv only)
	  -S TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT	 float, default 0.9, Fraction of data to be used for training the model
	  --pre_split           	Bool, default false, Indicator variable, which tells if the train test data is splitted beforehand
	  --error_metric ERROR_METRIC	string, default 'mse', Error metric to be used. options are  'mse'(mean squared error), 'mae' (mean absolute error) and 'rmse'
	  -l LOSS, --loss LOSS  	Loss function to be optimized. Default is 'ls' least square loss, Other options are: 'lad': Least Absolute Devaition
		
	  -r LEARNING_RATE, --learning_rate LEARNING_RATE	Learning rate: learning rate shrinks the contribution of each tree by `lrate`.
	  -n N_ESTIMATORS, --n_estimators N_ESTIMATORS	The number of boosting stages to perform
	  --subsample SUBSAMPLE	The fraction of samples to be used for fitting the individual base learners.
	  -d MAX_DEPTH, --max_depth MAX_DEPTH	 Maximum depth upto which the trees should grow.
	  --max_features MAX_FEATURES	 Number of features to be included in the final model
	  --min_samples_split MIN_SAMPLES_SPLIT	The minimum number of samples required to split an internal node
	  --min_samples_leaf MIN_SAMPLES_LEAF	The minimum number of samples required to be at a leaf  node


## Random Forest -- Command Line Usage

	  -h, --help            	show this help message and exit
	  -V, --verbose         	Flag argument, Enable verbose output.
	  --run_test           	 Flag argument which runs the tests
	  --train               	Boolean flag argument for running the training over data
	  --test                	Boolean flag argument for running the testing over pretrained model. Input model file is required
	  -T, --train_test      	Boolean flag argument for running training and testing simultaneously
	  -P, --predict         	Boolean flag argument for running the predictions on a pretrained model. Model input file is neccessary, Returns predicted values. and stores them in result file
	  --feature_imp         	 string, *.csv file, deafult None, Return feature importance
	  --result_path RESULT_PATH	string,*.csv file, deafult None, Result output file path.
	  --input_model INPUT_MODEL	string, deafult None, Input Model Address
	  --model_output_path MODEL_OUTPUT_PATH	 string, deafult None, Output Model File Path
	  -X X                  	string, *.csv file , default None, Train Test X Data path(csv only, if the data is not splitted beforehand
	  -Y Y                  	string, *.csv file , default None, Train Test Y Data path(csv only), if the data is not  splitted beforehand
	  -X_train X_TRAIN      	string, *.csv file , default None, Regressor(X) data path (csv only)
	  -Y_train Y_TRAIN      	string, *.csv file , default None, Label(Y) data path (csv only)
	  -X_test X_TEST        	string, *.csv file , default None , Regressor(X) data path (csv only)
	  -Y_test Y_TEST        	string, *.csv file , default None, Label(Y) data path (csv only)
	  -X_predict X_PREDICT  	string, *.csv file , default None Regressor(X) data path (csv only)
	  -S TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT	 float, default 0.9, Fraction of data to be used for training the model
	  --pre_split           	Bool, default false, Indicator variable, which tells if the train test data is splitted beforehand
	  --error_metric ERROR_METRIC	string, default 'mse', Error metric to be used. options are  'mse'(mean squared error), 'mae' (mean absolute error) and 'rmse'
	  -l LOSS, --loss LOSS  	Loss function to be optimized. Default is 'ls' least square loss, Other options are: 'lad': Least Absolute Devaition
		
	  --n_estimators	                        integer, optional (default=10) The number of trees in the forest
	  --criterion CRITERION	 string, optional {'mse', 'mae'} (default='mse'). The function to measure the quality of a split.
	  --max_features MAX_FEATURES	 int, float, string or None, optional (default='auto')
	  --max_depth MAX_DEPTH	integer or None, optional (default=None) The maximum depth of the tree.
	  --min_samples_split MIN_SAMPLES_SPLIT	int, float, optional (default=2) The minimum number of samples required to split an internal node:
	  --min_samples_leaf MIN_SAMPLES_LEAF	int, float, optional (default=1) The minimum number of samples required to be at a leaf node:
	  --max_leaf_nodes MAX_LEAF_NODES	int or None, optional (default=None) Grow trees with  ``max_leaf_nodes`` in best-first fashion.
	  --oob_score          	 bool, optional (default=False)whether to use out-of- bag samples to estimatethe R^2 on unseen data.
	  --n_jobs N_JOBS       	integer, optional (default=1)The number of jobs to run in parallel for both `fit` and `predict`. If -1, then the number of jobs is set to the number of cores.


## SV Classification -- Command Line Usage

	Argument	Description
	 --verbose	verbose : bool, optional, default False
	  --run_test            	Runs test
	  --train               	Boolean argument for running the training over data
	  --test                	Boolean argument for running the testing over pretrained model. Input model file is required
	  -T, --train_test     	 Boolean argument for running training and testing simultaneously
	  -P, --predict         	Boolean argument for running the predictions on a pretrained model. Model input file is neccessary, Returns predicted values. and stores them in result file
	  --result_path RESULT_PATH	Result output file path.
	  --input_model INPUT_MODEL	                        Input Model Address
	  --model_output_path MODEL_OUTPUT_PATH	                        Output Model File Path
	  -X X                  	Train Test X Data path(csv only, if the data is not splitted beforehand
	  -Y Y                  	Train Test Y Data path(csv only), if the data is not splitted beforehand
	  -X_train X_TRAIN      	Regressor(X) data path (csv only)
	  -Y_train Y_TRAIN      	Label(Y) data path (csv only)
	  -X_test X_TEST        	Regressor(X) data path (csv only)
	  -Y_test Y_TEST        	Label(Y) data path (csv only)
	  -X_predict X_PREDICT  	Regressor(X) data path (csv only)
	  -S TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT	                        Fraction of data to be used for training the model
	  --pre_split           	Boolean Indicator variable, which tells if the train test data is splitted beforehand
	  --error_metric ERROR_METRIC	                        Error metric to be used. Available options are 'clf_report'(Classfication Report: Build a text report showing the main classification metrics ), 'conf_mat' (Confusion Matrix: Compute  confusion matrix to  evaluate the accuracy of a classification), 'f1_score'  (Compute the F1 score, also known as balanced F-score Support for each class) 'precision' (Compute the precision) 'recall' (Compute the recall), 'prfs' (Precision Recall FScore and Support for each class) 
	  -C C, --C C           	float, optional (default=1.0) Penalty parameter C of the error term.
	  -k KERNEL, --kernel KERNEL	string, optional (default='rbf') Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
	  -d DEGREE, --degree DEGREE	int, optional (default=3) Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
	  --gamma GAMMA         	float, optional (default='auto') Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will be used instead.
	  --coef0 COEF0         	float, optional (default=0.0) Independent term in kernel function. It is only significant in 'poly' and  'sigmoid'.
	  --probability         	boolean, optional (default=False) Whether to enable probability estimates. This must be enabled prior to  calling `fit`, and will slow down that method.
	  --no_shrinking        	boolean, optional (default=True) Whether to use the shrinking heuristic.
	  --tol TOL             	float, optional (default=1e-3) Tolerance for stopping criterion.
	  --max_iter MAX_ITER   	int, optional (default=-1) Hard limit on iterations within solver, or -1 for no limit.


## SV Regression -- Command Line Usage

	Argument	Description
	 --verbose	verbose : bool, optional, default False
	  --run_test            	Runs test
	  --train               	Boolean argument for running the training over data
	  --test                	Boolean argument for running the testing over pretrained model. Input model file is required
	  -T, --train_test     	 Boolean argument for running training and testing simultaneously
	  -P, --predict         	Boolean argument for running the predictions on a pretrained model. Model input file is neccessary, Returns predicted values. and stores them in result file
	  --result_path RESULT_PATH	Result output file path.
	  --input_model INPUT_MODEL	                        Input Model Address
	  --model_output_path MODEL_OUTPUT_PATH	                        Output Model File Path
	  -X X                  	Train Test X Data path(csv only, if the data is not splitted beforehand
	  -Y Y                  	Train Test Y Data path(csv only), if the data is not splitted beforehand
	  -X_train X_TRAIN      	Regressor(X) data path (csv only)
	  -Y_train Y_TRAIN      	Label(Y) data path (csv only)
	  -X_test X_TEST        	Regressor(X) data path (csv only)
	  -Y_test Y_TEST        	Label(Y) data path (csv only)
	  -X_predict X_PREDICT  	Regressor(X) data path (csv only)
	  -S TRAIN_TEST_SPLIT, --train_test_split TRAIN_TEST_SPLIT	                        Fraction of data to be used for training the model
	  --pre_split           	Boolean Indicator variable, which tells if the train test data is splitted beforehand
	  --error_metric ERROR_METRIC	                        Error metric to be used. Available options are 'clf_report'(Classfication Report: Build a text report showing the main classification metrics ), 'conf_mat' (Confusion Matrix: Compute  confusion matrix to  evaluate the accuracy of a classification), 'f1_score'  (Compute the F1 score, also known as balanced F-score Support for each class) 'precision' (Compute the precision) 'recall' (Compute the recall), 'prfs' (Precision Recall FScore and Support for each class) 
	  -C C, --C C           	float, optional (default=1.0) Penalty parameter C of the error term.
	--epsilon EPSILON, -E EPSILON	float, optional (default=0.1)Epsilon in the epsilon- SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
	  -k KERNEL, --kernel KERNEL	string, optional (default='rbf') Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable.
	  -d DEGREE, --degree DEGREE	int, optional (default=3) Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
	  --gamma GAMMA         	float, optional (default='auto') Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will be used instead.
	  --coef0 COEF0         	float, optional (default=0.0) Independent term in kernel function. It is only significant in 'poly' and  'sigmoid'.
	  --no_shrinking        	boolean, optional (default=True) Whether to use the shrinking heuristic.
	  --tol TOL             	float, optional (default=1e-3) Tolerance for stopping criterion.
	  --max_iter MAX_ITER   	int, optional (default=-1) Hard limit on iterations within solver, or -1 for no limit.

