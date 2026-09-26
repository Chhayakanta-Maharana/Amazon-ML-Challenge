import os
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

class EntityMatcherModel:
    def __init__(self, random_state=42):
        self.model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.07,
            max_depth=7,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=random_state
        )
        self.best_threshold = 0.50
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
        
    def predict(self, X, threshold=None):
        thresh = threshold if threshold is not None else self.best_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "threshold": self.best_threshold
        }, filepath)
        
    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.best_threshold = data["threshold"]
        return instance
