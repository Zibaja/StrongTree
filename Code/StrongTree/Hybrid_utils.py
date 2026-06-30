
import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset():
    """Class representing a dataset loaded from a csv file.
    csv file is after rule mining"""
    def __init__(self, dataset_name, X, y, features, prediction):
        self.name = dataset_name
        self.X = X
        self.features = features
        self.y = y
        self.prediction = prediction
        self.preprocessed = False
    

    @classmethod
    def from_csv(cls, fname, dataset_name ):
        """
        Load a dataset from a csv file. The csv file must contain n_samples+1 rows, each with n_features+1
        columns. The last column of each sample is its prediction class, and the first row of the file
        contains the feature names and prediction class name.
        
        Parameters
        ----------
        fname : str
            File name of the csv data file
        
        Returns
        -------
        X : array-like, shape = [n_samples, n_features]
            The sample data

        y : array-line, shape = [n_samples]
            The target values for the sample data
        
        features : list
            A list of strings of length n_features. Specifies the names of each of the features.

        prediction_name : str
            The name of the prediction class
        """
        df = pd.read_csv(fname)
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        features = df.columns[:-1].tolist()
        prediction = df.columns[-1]
        return cls(dataset_name, X, y, features, prediction)


    def train_test_split(self, train_proportion, random_state):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=1.0 - train_proportion, shuffle=True,random_state=random_state)
        X_dict = {'train': X_train, 'test':X_test }
        y_dict = {'train': y_train, 'test':y_test }
        return X_dict, y_dict
    

    def pre_process (self):
        """This method apply all modifications regarding demographic subgroups
        """
        raise NotImplementedError("This method is not implemented yet")
        


    def split_data_as_dict(self, splits, random_state_param=42):
        """This method split data to train and test set after preprocessing
        # it was called get_data_norulemining in hybridCORELS
        Args:
            splits (dict): example {"train" : 0.8, "test" : 0.2}
            random_state_param (int, optional):  Defaults to 42.

        Returns:
            dict: the output is X={'train':X_train,'test': X_test}
        and y={'train':y_train,'test': y_test}
        """

        # Generate splits
        assert len(splits) <= 3, "We only support splitting the data to up to 3 folds"
        split_names = list(splits.keys())
        split_ratios = list(splits.values())
        assert np.sum(split_ratios) == 1, "The split ratios must sum up to one"
        X_dict = {}
        y_dict = {}
        X_1, X_2, y_1, y_2 = train_test_split(self.X, self.y, train_size=split_ratios[0],
                                            shuffle=True, random_state=random_state_param)
        X_dict[split_names[0]] = X_1
        y_dict[split_names[0]] = y_1
        if len(splits) == 2:
            X_dict[split_names[1]] = X_2
            y_dict[split_names[1]] = y_2
        else:
            sub_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
            X_2, X_3, y_2, y_3 = train_test_split(X_2, y_2, train_size=sub_ratio,
                                            shuffle=True, random_state=random_state_param)
            X_dict[split_names[1]] = X_2
            y_dict[split_names[1]] = y_2
            X_dict[split_names[2]] = X_3
            y_dict[split_names[2]] = y_3
        return X_dict, y_dict, self.features, self.prediction
    


    def split_data_as_dict_withsize(self, splits, random_state_param=42):
        """This method split data to train and test set after preprocessing
        # it was called get_data_norulemining in hybridCORELS
        Args:
            splits (dict): example {"train" : 0.8, "test" : 0.2}
            random_state_param (int, optional):  Defaults to 42.

        Returns:
            dict: the output is X={'train':X_train,'test': X_test}
        and y={'train':y_train,'test': y_test}
        """

        # Generate splits
        assert len(splits) <= 3, "We only support splitting the data to up to 3 folds"
        split_names = list(splits.keys())
        split_ratios = list(splits.values())
        assert np.sum(split_ratios) <= self.X.shape[0], "The split size must be less than or equal to the number of datapoints"
        X_dict = {}
        y_dict = {}
        X_1, X_2, y_1, y_2 = train_test_split(self.X, self.y, train_size=split_ratios[0],
                                            shuffle=True, random_state=random_state_param)
        X_dict[split_names[0]] = X_1
        y_dict[split_names[0]] = y_1
        if len(splits) == 2:
            X_dict[split_names[1]] = X_2
            y_dict[split_names[1]] = y_2
        else:
            # sub_ratio = split_ratios[1] / (split_ratios[1] + split_ratios[2])
            X_2, X_3, y_2, y_3 = train_test_split(X_2, y_2, train_size=split_ratios[1], test_size= split_ratios[2],
                                            shuffle=True, random_state=random_state_param)
            X_dict[split_names[1]] = X_2
            y_dict[split_names[1]] = y_2
            X_dict[split_names[2]] = X_3
            y_dict[split_names[2]] = y_3
        return X_dict, y_dict, self.features, self.prediction
    

    def to_df(self):
        return pd.DataFrame(self.X, columns=self.features)
    
    
    def to_df_from_dict(self, X_dict):
        df_X = {}
        for key, val in X_dict.items():
            df_X[key] = pd.DataFrame(val, columns=self.features)
        return df_X
    

def get_node_status(hybridDT,n):
    '''
    This function give the status of a given node in a tree. By status we mean whether the node
        1- is pruned? i.e., we have made a prediction at one of its ancestors
        2- is a branching node? If yes, what feature do we branch on
        3- is a leaf? If yes, what is the prediction at this node?

    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :param n: A valid node index in the tree
    :return: pruned, branching, selected_feature, leaf, value

    pruned=1 iff the node is pruned
    branching = 1 iff the node branches at some feature f
    selected_feature: The feature that the node branch on
    leaf = 1 iff node n is a leaf in the tree
    value: if node n is a leaf, value represent the prediction at this node
    '''
    tree = hybridDT.tree
    # mode = grb_model.mode
    pruned = False
    branching = False
    leaf = False
    value = None
    selected_feature = None

    p_sum = 0
    for m in tree.get_ancestors(n):
        p_sum = p_sum + hybridDT.p[m]
    if hybridDT.p[n] > 0.5:  # leaf
        leaf = True
        # if mode == "regression":
        #     value = beta[n, 1]
        # elif mode == "classification":
        for k in hybridDT.labels:
            if hybridDT.beta[n, k] > 0.5:
                value = k
    elif p_sum == 1:  # Pruned
        pruned = True

    if n in tree.Nodes:
        if (pruned == False) and (leaf == False):  # branching
            for f in hybridDT.features:
                if hybridDT.b[n, f] > 0.5:
                    selected_feature = f
                    branching = True

    return pruned, branching, selected_feature, leaf, value



def print_tree(hybridDT):
    '''
    This function print the derived tree with the branching features and the predictions asserted for each node
    :param grb_model: the gurobi model solved to optimality (or reached to the time limit)
    :param b: The values of branching decision variable b
    :param beta: The values of prediction decision variable beta
    :param p: The values of decision variable p
    :return: print out the tree in the console
    '''
    tree = hybridDT.tree
    for n in tree.Nodes + tree.Leaves:
        pruned, branching, selected_feature, leaf, value = get_node_status(hybridDT, n)
        print('#########node ', n)
        if pruned:
            print("pruned")
        elif branching:
            print(selected_feature)
        elif leaf:
            print('leaf {}'.format(value))


def get_predicted_value(hybridDT, local_data, i):
    '''
    This function returns the predicted value for a given datapoint
    :param grb_model: The gurobi model we solved
    :param local_data: The dataset we want to compute accuracy for
    :param b: The value of decision variable b
    :param beta: The value of decision variable beta
    :param p: The value of decision variable p
    :param i: Index of the datapoint we are interested in
    :return: The predicted value for datapoint i in dataset local_data
    '''
    tree = hybridDT.tree
    current = 1 # we start traversing through the tree until we reach a leaf

    while True:
        pruned, branching, selected_feature, leaf, value = get_node_status(hybridDT, current)
        if leaf:
            return value
        elif branching:
            if local_data.at[i, selected_feature] == 1:  # going right on the branch
                current = tree.get_right_children(current)
            else:  # going left on the branch
                current = tree.get_left_children(current)


def get_acc(y, y_hat):
    acc = np.mean (y==y_hat)
    return acc

