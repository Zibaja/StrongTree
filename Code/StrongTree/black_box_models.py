''' This file centralizes declarations for the different black-boxes used in our experiments
The main object to be used is the BlackBox class 
Careful: if your version of sklearn is too old you might get errors! '''

# Create and train the hyperparameter-optimized black-box model
from hyperopt import tpe, hp, Trials
from hyperopt.fmin import fmin
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# hyperopt-sklearn declarations
def _forest_classifier_criterion(name: str):
    """
    Declaration of search space 'criterion' parameter for
     random forest classifier
     extra trees classifier
    """
    return hp.choice(name, ["gini", "entropy"])

def _forest_class_weight(name: str):
    """
    Declaration of search space 'class_weight' parameter for
     random forest classifier
     extra trees classifier
    """
    return hp.choice(name, ["balanced", "balanced_subsample", None])

def _forest_n_estimators(name: str):
    """
    Declaration search space 'n_estimators' parameter
    """
    return hp.qloguniform(name, np.log(9.5), np.log(500.5), 1)#np.log(3000.5), 1)

def _forest_max_depth(name: str):
    """
    Declaration search space 'max_depth' parameter
    """
    return hp.pchoice(name, [
        (0.7, None),  # most common choice.
        (0.1, 2),  # try some shallow trees.
        (0.1, 3),
        (0.1, 4),
    ])

def _forest_min_samples_split(name: str):
    """
    Declaration search space 'min_samples_split' parameter
    """
    return hp.pchoice(name, [
        (0.95, 2),  # most common choice
        (0.05, 3),  # try minimal increase
    ])

def _forest_min_samples_leaf(name: str):
    """
    Declaration search space 'min_samples_leaf' parameter
    """
    return hp.choice(name, [
        1,  # most common choice.
        hp.qloguniform(name + ".gt1", np.log(1.5), np.log(50.5), 1)
    ])

def _forest_min_weight_fraction_leaf(name: str):
    """
    Declaration search space 'min_weight_fraction_leaf' parameter
    """
    return 0.0

def _forest_max_features(name: str):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.pchoice(name, [
        (0.2, "sqrt"),  # most common choice.
        (0.1, "log2"),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + ".frac", 0., 1.))
    ])

def _forest_max_leaf_nodes(name: str):
    """
    Declaration search space 'max_leaf_nodes' parameter
    """
    return hp.pchoice(name, [
        (0.85, None),  # most common choice
        (0.05, 5),
        (0.05, 10),
        (0.05, 15),
    ])

def _forest_min_impurity_decrease(name: str):
    """
    Declaration search space 'min_impurity_decrease' parameter
    """
    return hp.pchoice(name, [
        (0.85, 0.0),  # most common choice
        (0.05, 0.01),
        (0.05, 0.02),
        (0.05, 0.05),
    ])

def _forest_bootstrap(name: str):
    """
    Declaration search space 'bootstrap' parameter
    """
    return hp.choice(name, [True, False])

def _forest_random_state(name: str):
    """
    Declaration search space 'random_state' parameter
    """
    return hp.randint(name, 5)

def _weight_boosting_algorithm(name: str):
    """
    Declaration search space 'algorithm' parameter
    """
    return hp.choice(name, ["SAMME", "SAMME.R"])
    
def _weight_boosting_n_estimators(name: str):
    """
    Declaration search space 'n_estimators' parameter
    """
    return hp.qloguniform(name, np.log(10.5), np.log(1000.5), 1)

def _weight_boosting_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.lognormal(name, np.log(0.01), np.log(10.0))

def _gb_clf_loss(name: str):
    """
    Declaration search space 'loss' parameter for _gb classifier
    """
    return hp.choice(name, ["log_loss", "exponential"])

def _gb_learning_rate(name: str):
    """
    Declaration search space 'learning_rate' parameter
    """
    return hp.lognormal(name, np.log(0.01), np.log(10.0))

def _gb_n_estimators(name: str):
    """
    Declaration search space 'n_estimators' parameter
    """
    return hp.qloguniform(name, np.log(10.5), np.log(1000.5), 1)

def _gb_criterion(name: str):
    """
    Declaration search space 'criterion' parameter
    """
    return hp.choice(name, ['friedman_mse', 'squared_error'])

def _gb_min_samples_split(name: str):
    """
    Declaration search space 'min_samples_split' parameter
    """
    return hp.pchoice(name, [
        (0.95, 2),  # most common choice
        (0.05, 3),  # try minimal increase
    ])

def _gb_min_samples_leaf(name: str):
    """
    Declaration search space 'min_samples_leaf' parameter
    """
    return hp.choice(name, [
        1,  # most common choice.
        hp.qloguniform(name + ".gt1", np.log(1.5), np.log(50.5), 1)
    ])

def _gb_min_weight_fraction_leaf(name: str):
    """
    Declaration search space 'min_weight_fraction_leaf' parameter
    """
    return 0.0

def _gb_max_depth(name: str):
    """
    Declaration search space 'max_depth' parameter
    """
    return hp.pchoice(name, [
        (0.1, 2),
        (0.7, 3),  # most common choice.
        (0.1, 4),
        (0.1, 5),
    ])

def _gb_min_impurity_decrease(name: str):
    """
    Declaration search space 'min_impurity_decrease' parameter
    """
    return hp.pchoice(name, [
        (0.85, 0.0),  # most common choice
        (0.05, 0.01),
        (0.05, 0.02),
        (0.05, 0.05),
    ])

def _gb_max_features(name: str):
    """
    Declaration search space 'max_features' parameter
    """
    return hp.pchoice(name, [
        (0.2, "sqrt"),  # most common choice.
        (0.1, "log2"),  # less common choice.
        (0.1, None),  # all features, less common choice.
        (0.6, hp.uniform(name + ".frac", 0., 1.))
    ])

def _gb_max_leaf_nodes(name: str):
    """
    Declaration search space 'max_leaf_nodes' parameter
    """
    return hp.pchoice(name, [
        (0.85, None),  # most common choice
        (0.05, 5),
        (0.05, 10),
        (0.05, 15),
    ])


def correct_names(best, to_int_params):
    to_rename = []
    for p in best.keys():
        if '.' in p:
            p_split = p.split('.')
            p_new = p_split[0]
            to_rename.append([p, p_new])
    for ppnew in to_rename:
        p = ppnew[0]
        p_new = ppnew[1]
        best[p_new] = best[p]
        best.pop(p)
    for p in to_int_params:
        if not best[p] is None:
            best[p] = int(best[p])

    return best

# Here is the main object
class BlackBox:
    def __init__(self, bb_type, verbosity=False, random_state_value=42, n_iter=100, time_limit=None, X_val=None, y_val=None, sample_weights_val=None):
        '''
        bb_type: str, type of black-box model to be trained
            Supported BB types: 
                - "random_forest"
                - "ada_boost"
                - "gradient_boost"
        
        verbosity: bool (default False)
            To print useful information: True
            To print nothing: False
        
        random_state_value: int (default 42)
            Seed to initialize all random processes.
            Fix a value for reproducibility.

        n_iter: int (default 100)
            Number of iterations for Hyperopt's Fmin function.

        time_limit: None or int (default None)
            Maximum training time (approx. time of call to fit())
            None for no limit or int value in #seconds.        

        X_val, y_val: 
            Optional validation set (see .fit method)
        '''
        self.random_state_value = random_state_value
        self.verbosity = verbosity
        self.bb_type = bb_type
        self.n_iter = n_iter
        self.time_limit = time_limit
        self.is_fitted = False
        if X_val is None or y_val is None:
            self.provided_validation_data = False 
        else:
            self.provided_validation_data = True 
            self.X_val = X_val
            self.y_val = y_val 
            self.sample_weights_validation = sample_weights_val

    def fit(self, X, y, sample_weight=None):
        """
        Runs Hyperparameters optimization using the provided data.
        
        There are two possibilities depending on whether the user provided validation data while creating this object:

        1 - No validation data provided when creating this object(X_val and y_val left to their default None value).
        In this case, the validation set is automatically done by our method.
        and the eventual sample_weights are applied both during training and for model evaluation (weighted accuracy)
        CAREFUL the final model (with tuned hyperparameters) is then trained on the entire data (X,y)

        2 - Validation data provided when creating this object (through X_val and y_val). 
        In this case, the hyperparameter search optimizes the (unweighted) validation accuracy
        and only uses the training data (X and y given to this method) only to fit the model 
        (the validation data X_val and y_val is only used to determine the best hyperparameters)
        """
        # print("min sw = ", np.min(sample_weight), ", max sw = ", np.max(sample_weight))
        # define a validation set using one third of the data (and keep track of indices for the sample weights)
        if not self.provided_validation_data:
            if self.verbosity:
                print("Performing a validation split to determine the best hyperparameters and avoid overfitting.")
            indices = np.arange(y.size)
            (
                X_train,
                X_val,
                y_train,
                y_val,
                indices_train,
                indices_val,
            ) = train_test_split(X, y, indices, test_size=(1/3))


            if not sample_weight is None:
                sample_weight_train = sample_weight[indices_train]
                sample_weight_val = sample_weight[indices_val]
            else:
                sample_weight_train = None
                sample_weight_val = None
        else:
            if self.verbosity:
                print("Using provided validation data.")
            X_train = X
            y_train = y
            sample_weight_train = sample_weight
            sample_weight_val = self.sample_weights_validation
            X_val = self.X_val
            y_val = self.y_val

        # define the BB type
        if self.bb_type == "random_forest":
            # retrieve classifier object constructor
            classifier_wrapper = RandomForestClassifier
            
            # define grid as is done inside hyperopt-sklearn
            params={'n_estimators':_forest_n_estimators('n_estimators'),
                    'max_depth':_forest_max_depth('max_depth'),
                    'min_samples_leaf':_forest_min_samples_leaf('min_samples_leaf'),
                    'min_samples_split':_forest_min_samples_split('min_samples_split'),
                    'min_weight_fraction_leaf':_forest_min_weight_fraction_leaf('min_weight_fraction_leaf'),
                    'max_features':_forest_max_features('max_features'),
                    'max_leaf_nodes':_forest_max_leaf_nodes('max_leaf_nodes'),
                    'min_impurity_decrease':_forest_min_impurity_decrease('min_impurity_decrease'),
                    'bootstrap':_forest_bootstrap('bootstrap'),
                    'criterion':_forest_classifier_criterion('criterion'),
                    'class_weight':_forest_class_weight('class_weight')
            }
            to_int_params = ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split']

        elif self.bb_type == "ada_boost":
            # retrieve classifier object constructor
            classifier_wrapper = AdaBoostClassifier

            # define grid as is done inside hyperopt-sklearn
            params={'algorithm':_weight_boosting_algorithm('algorithm'),
            'n_estimators':_weight_boosting_n_estimators('n_estimators'),
            'learning_rate':_weight_boosting_learning_rate('learning_rate')                    
            }
            to_int_params = ['n_estimators']

        elif self.bb_type == "gradient_boost":
            # retrieve classifier object constructor
            classifier_wrapper = GradientBoostingClassifier
            # define grid as is done inside hyperopt-sklearn
            params={'loss': _gb_clf_loss('loss'),
            'learning_rate':_gb_learning_rate('learning_rate'),
            'n_estimators':_gb_n_estimators('n_estimators'),
            'criterion':_gb_criterion('criterion'),
            'min_samples_split':_gb_min_samples_split('min_samples_split'),
            'min_samples_leaf':_gb_min_samples_leaf('min_samples_leaf'),
            'min_weight_fraction_leaf':_gb_min_weight_fraction_leaf('min_weight_fraction_leaf'),
            'max_depth':_gb_max_depth('max_depth'),
            'min_impurity_decrease':_gb_min_impurity_decrease('min_impurity_decrease'),
            'max_features':_gb_max_features('max_features'),
            'max_leaf_nodes':_gb_max_leaf_nodes('max_leaf_nodes')
            }
            to_int_params = ['min_samples_leaf', 'n_estimators']
        
        # find best params
        def objective(local_params):
            global n_evals
            local_params["random_state"] = self.random_state_value
            #print(local_params)
            for p in to_int_params:
                if not local_params[p] is None:
                    local_params[p] = int(local_params[p])
            #print(local_params)
            model=classifier_wrapper(**local_params)
            model.fit(X_train,y_train,sample_weight=sample_weight_train)
            return (1.0 - model.score(X_val, y_val, sample_weight=sample_weight_val)) # minimize validation error
        
        trials = Trials()

        # Old way: single call to FMin, all iters have global timeout together
        #best=fmin(fn=objective, space=params, algo=tpe.suggest, max_evals=self.n_iter,
        #          rstate=np.random.default_rng(self.random_state_value), show_progressbar=self.verbosity, 
        #          return_argmin=False, timeout=self.time_limit, trials=trials)

        # New way: individual calls to FMin, each iter has its own timeout
        rstate_object = np.random.default_rng(self.random_state_value)
        for iter_id in range(1,self.n_iter+1):
            best=fmin(fn=objective, space=params, algo=tpe.suggest, max_evals=iter_id,
                    rstate=rstate_object, show_progressbar=self.verbosity, 
                    return_argmin=False, timeout=(self.time_limit / self.n_iter), trials=trials)

                  
        self.trials_details = trials.trials
        self.n_evals = len(trials.results)
        
        best = correct_names(best, to_int_params)
        
        best["random_state"] = self.random_state_value

        if self.verbosity:
            print("Best Hyperparameters: ", best)

        self.black_box_model = classifier_wrapper(**best)
        self.black_box_model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        
        return self

    def predict(self, X):
        return self.black_box_model.predict(X)
    
    def score(self, X, y):
        return self.black_box_model.score(X, y)

    def __str__(self):
        return str(self.black_box_model)


    def __sklearn_is_fitted__(self):
        return self.is_fitted


    def save(self, fname):
        """
        Save the black box to a file, using python's pickle module.

        Parameters
        ----------
        fname : string
            File name to store the model in
        
        Returns
        -------
        self : obj
        """
        import pickle

        with open(fname, "wb") as f:
            pickle.dump(self, f)

        return self

    
    def load(self, fname):
        """
        Load a black box from a file, using python's pickle module.
        
        Parameters
        ----------
        fname : string
            File name to load the rulelist from
        
        Returns
        -------
        self : obj
        """
        import pickle
        with open(fname, "rb") as f:
            loaded_object = pickle.load(f)
        if type(loaded_object) != BlackBox:
            raise TypeError("Loaded object of type %s from file %s, expected <class 'black_box_models.BlackBox'>" %(type(loaded_object), fname))
        else:
            return loaded_object