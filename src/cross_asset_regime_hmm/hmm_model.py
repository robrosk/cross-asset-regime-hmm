import numpy as np
from hmmlearn.hmm import GaussianHMM

class RegimeHMM:
    def __init__(self, n_states=3, covariance_type="full", n_iter=500, random_state=42):
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        return self

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def means_(self):
        return self.model.means_

    @property
    def covars_(self):
        return self.model.covars_

    @property
    def transmat_(self):
        return self.model.transmat_
