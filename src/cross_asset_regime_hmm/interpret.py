import numpy as np

def label_states(means: np.ndarray, return_idx: int = 0) -> dict[int, str]:
    mu = means[:, return_idx]
    bull = int(mu.argmax())
    bear = int(mu.argmin())
    volatile = int(({0,1,2} - {bull, bear}).pop())
    return {bull: "Bull", bear: "Bear", volatile: "Volatile"}

