import numpy as np
import torch
from sklearn import linear_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification


class Model:
    def __init__(self):
        self.name = "Base Model"

    def train(self, features, labels):
        pass

    def predict(self, features):
        pass

    def __repr__(self):
        return f"{self.name}"


class RandomPredictor(Model):
    def __init__(self):
        super().__init__()
        self.name = "Random Predictor"

    def train(self, features, labels):
        super().train(features, labels)

    def predict(self, features):
        super().predict(features)
        return np.random.choice([1, -1], size=features.shape[0])


class LinearRegression(Model):
    def __init__(self):
        super().__init__()
        self.name = f"Linear Regression"
        self.model = linear_model.LinearRegression()

    def train(self, features, labels):
        super().train(features, labels)
        self.model.fit(features, labels)

    def predict(self, features):
        super().predict(features)
        y_pred = self.model.predict(features)
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        return y_pred


class LogisticRegression(LinearRegression):
    def __init__(self):
        super().__init__()
        self.name = f"Logistic Regression"
        self.model = linear_model.LogisticRegression(C=1e5, max_iter=10000)


class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.name = f"Ridge Regression"
        self.alpha = alpha
        self.model = linear_model.Ridge(self.alpha, max_iter=10000)

    def __repr__(self):
        return f"{self.name} ({self.alpha})"


class LassoRegression(LinearRegression):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.name = f"Lasso Regression"
        self.alpha = alpha
        self.model = linear_model.Lasso(self.alpha, max_iter=10000)

    def __repr__(self):
        return f"{self.name} ({self.alpha})"


def choose_ridge_lasso_hyper_parameters(model_cls, features, labels, min_alpha_pow=-5, max_alpha_pow=5, fraction=0.9):
    split_idx = int(fraction * features.shape[0])

    best_accuracy = (1.0, 0.5)
    for alpha_pow in tqdm(range(min_alpha_pow, max_alpha_pow), desc=f"Choosing hyper parameter..."):
        alpha = 10 ** alpha_pow
        model = model_cls(alpha=alpha)
        model.train(features[:split_idx], labels[:split_idx])
        predicted = model.predict(features[split_idx:])
        accuracy = (predicted == labels[split_idx:]).mean()
        if accuracy > best_accuracy[-1]:
            best_accuracy = (model.alpha, accuracy)

    return model_cls(alpha=best_accuracy[0])
