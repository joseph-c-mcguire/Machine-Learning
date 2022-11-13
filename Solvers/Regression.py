import torch

class Regressor():
    """
    This is a class for regression

    ...

    Attributes
    __________
    X : N x D array_like
        This is the full design matrix
    Y : N x M array_like
        This is the full target matrix
    Methods
    _______

    """
    def __init__(self,X,Y, solver = None):
        """
        Initializes the class Regressor

        ...

        Parameters
        __________
        X : N x D array_like
            This is the full design matrix
        Y : N x D array_like
            This is the full target matrix

        """

        self.X = X
        self.Y = Y
        self.solver = solver
    def impute(self, X,method = None):
        """
        Imputes the data given based on column statistics, or constant value

        ...

        Parameters
        __________
        X : N x D array_like
            This is the data to be imputed
        method : {"mean", "median", "mode"} or float
            This is the method of imputation, or if float the value to replace the missing values with

        Returns
        _______
        X' : N x D array_like
            This is the data post imputation

        ToDo
        ____
        - Allow for array_likes in the method section
        - Allow for linear or model fits for imputation
        """

        imputation = {
            "mean" : torch.mean,
            "median" : torch.median,
            "mode" : torch.mode
        }
        if method is None:
            method = "mean"
        if type(method) == (int or float):
            impute = torch.ones(X.shape[1]) * method
        else:
            impute = imputation[method](X, 0)
        X = torch.clone(X)
        for i in range(X.shape[1]):
            X[:,i] = torch.nan_to_num(impute[i])

        return X
