import numpy as np
import matplotlib.pyplot as plt
import time
from MLSDeformation import mls_affine_transform_pre_calc, calculate_new_v

# Create grid
xgrid = np.linspace(0, 20, 150)
ygrid = np.linspace(0, 20, 150)
grid = []
for i in xgrid:
    for j in ygrid:
        grid.append(np.array([i, j]))
grid = np.array(grid)
p = [[2, 2], [2, 10], [2, 18], [10, 2], [10, 10], [10, 18], [18, 2], [18, 10], [18, 18]]
q = [[2, 4], [4, 8], [1, 14], [10, 5], [10, 14], [10, 14], [16, 6], [18, 10], [19, 20]]

mls_affine_transform_pre_calc(p, grid, 1)
A_list, w, w_sum = mls_affine_transform_pre_calc(p, grid, 1)
time1 = time.time()
new_v = (calculate_new_v(q, A_list, w, w_sum, grid))
print(time.time() - time1)

print("done")
plt.scatter(grid[:, 0], grid[:, 1], s=5)
plt.scatter(new_v[:, 0], new_v[:, 1], s=5)
plt.scatter(np.array(p)[:, 0], np.array(p)[:, 1])
plt.scatter(np.array(q)[:, 0], np.array(q)[:, 1])

plt.show()
