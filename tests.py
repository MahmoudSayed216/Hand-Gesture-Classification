import numpy as np
from scipy import stats


ar = np.array(
[1,2,3,1,2,2,2,2,2,2,2,2,2,1,1,2,3,4,1,1,1,1,3,3,2]
)

window_size = 2
window = ar[-window_size: ]
print(window)