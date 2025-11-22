MEAN_PATIENT_PER_HOUR = [
    3.0, 2.8, 2.5, 2.2, 2.0, 2.3,
    3.0, 5.0, 7.5, 9.5, 10.5, 10.0,
    9.8, 9.2, 8.8, 7.8, 7.2, 6.6,
    6.0, 5.5, 4.8, 4.0, 3.5, 3.2
]
import numpy as np
from scipy.stats import gamma

rates = np.array(MEAN_PATIENT_PER_HOUR)

# 估计 gamma shape(k) 和 scale(θ)
shape, loc, scale = gamma.fit(rates, floc=0)
print(shape, scale)
