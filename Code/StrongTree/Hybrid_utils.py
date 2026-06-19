
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
    

    def to_df(self):
        return pd.DataFrame(self.X, columns=self.features)
    
    
    def to_df_from_dict(self, X_dict):
        df_X = {}
        for key, val in X_dict.items():
            df_X[key] = pd.DataFrame(val, columns=self.features)
        return df_X
