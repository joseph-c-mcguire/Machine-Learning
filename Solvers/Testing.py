from sklearn import datasets
import Regression
import torch

X,Y = datasets.make_regression(1000,20, n_targets=2)
X = torch.tensor(X)
Y = torch.tensor(Y)
reg = Regression.Regressor(X,Y)

print(reg.impute(X).shape)