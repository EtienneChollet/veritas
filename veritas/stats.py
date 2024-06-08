import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score


class LinearRegressionModel:
    def __init__(self, df, features, target, random_state=8608, remove_cols=[]):
        self.df = df
        self.features = features
        self.target = target
        self.random_state = random_state
        self.model = LinearRegression()
    
    def prepare_data(self):
        X = self.df[self.features]
        y = self.df[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.random_state)
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
    
    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
    
    def calculate_statistics(self):
        predictions = self.model.predict(self.df[self.features])
        residuals = self.df[self.target] - predictions
        residual_sum_of_squares = np.sum(residuals**2)
        deg_freedom = len(self.df[self.target]) - len(self.coefficients) - 1
        mse = residual_sum_of_squares / deg_freedom
        
        X_with_intercept = np.column_stack((np.ones(self.df[self.features].shape[0]), self.df[self.features]))
        var_beta = mse * np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept)).diagonal()
        se_beta = np.sqrt(var_beta)
        
        t_values = np.append(self.intercept, self.coefficients) / se_beta
        p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=deg_freedom)) for t in t_values]
        ci_lower = np.append(self.intercept, self.coefficients) - 1.96 * se_beta
        ci_upper = np.append(self.intercept, self.coefficients) + 1.96 * se_beta
        
        self.summary_df = pd.DataFrame({
            'Features': ['Intercept'] + list(self.df[self.features].columns),
            'Coefficients': np.append(self.intercept, self.coefficients),
            'Standard Errors': se_beta,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            't-values': t_values,
            'P-values': p_values,
            'P-value Interpretation': ['Significant' if p < 0.05 else 'Not significant' for p in p_values]
        })
    
    def evaluate_model(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        r_squared = self.model.score(self.X_test, self.y_test)
        print(f"Mean Squared Error: {mse}")
        print(f"R^2: {r_squared}")
    
    def get_summary(self):
        return self.summary_df
    

class MetricsCalculator:
    """
    Class to compute common segmentation metrics between two binary tensors.

    Attributes
    ----------
    true : torch.Tensor
        The ground truth binary tensor.
    pred : torch.Tensor
        The predicted binary tensor.
    """
    def __init__(self, true: torch.Tensor, pred: torch.Tensor, metric='dice'):
        """
        Initialize the calculator with the true and predicted tensors.

        Parameters
        ----------
        true : torch.Tensor
            The ground truth binary tensor.
        pred : torch.Tensor
            The predicted binary tensor.
        """
        assert true.shape == pred.shape, "Tensors must have the same shape"
        self.true = true
        self.pred = pred
        self.metric = metric

    def dice_score(self) -> float:
        """
        Compute the Dice coefficient.

        Returns
        -------
        float
            Dice coefficient as a float.
        """
        tp = (self.true * self.pred).sum()
        fp = (self.true < self.pred).sum()
        fn = (self.true > self.pred).sum()
        dice = (2 * tp) / ((2 * tp) + fp + fn)
        return dice.item()
        #intersection = (self.true * self.pred).sum()
        #total = self.true.sum() + self.pred.sum()
        #if total == 0:
        #    return 1.0
        #return (2 * intersection / total).item()

    def false_positive_ratio(self) -> float:
        """
        Compute the false positive ratio (FPR).

        Returns
        -------
        float
            False positive ratio as a float.
        """
        tp = (self.true * self.pred).sum()
        fp = (self.true < self.pred).sum()
        fn = (self.true > self.pred).sum()
        fpr= fp / (fp + tp + fn)
        return fpr
        #fp = ((self.pred == 1) & (self.true == 0)).sum()
        #tn = ((self.pred == 0) & (self.true == 0)).sum()
        #if fp + tn == 0:
        #    return 0.0
        #return (fp / (fp + tn)).item()

    def false_negative_ratio(self) -> float:
        """
        Compute the false negative ratio (FNR).

        Returns
        -------
        float
            False negative ratio as a float.
        """
        fn = ((self.pred == 0) & (self.true == 1)).sum()
        tp = ((self.pred == 1) & (self.true == 1)).sum()
        if fn + tp == 0:
            return 0.0
        return (fn / (fn + tp)).item()

    def cohens_kappa(self) -> float:
        """
        Compute Cohen's Kappa

        Returns
        -------
        float
            Cohen's Kappa as a float.
        """
        pred_flat = self.pred.cpu().numpy().flatten()
        true_flat = self.true.cpu().numpy().flatten()
        k = cohen_kappa_score(true_flat, pred_flat)
        return k

    def get_all(self):
        if self.metric == 'dice':
            dice = self.dice_score()
            fpr = self.false_positive_ratio()
            fnr = self.false_negative_ratio()
            out = torch.tensor([dice, fpr, fnr])
        elif self.metric == 'kappa':
            k = self.cohens_kappa()
            out = torch.tensor([k, 0, 0])
        return out