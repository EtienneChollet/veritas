import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
