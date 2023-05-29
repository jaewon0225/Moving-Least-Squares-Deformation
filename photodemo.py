import cv2
import numpy as np
import matplotlib.pyplot as plt
from MLSDeformation import mls_affine_transform_pre_calc, calculate_new_v


def make_grid(xdim, ydim):
    xgrid = np.linspace(0, xdim - 1, xdim)
    ygrid = np.linspace(0, ydim - 1, ydim)
    grid = []
    for i in xgrid:
        for j in ygrid:
            grid.append(np.array([i, j]))
    return np.array(grid)


# Inversion required since cv2.remap works in the opposite way
def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:, :, 1], I[:, :, 0] = np.indices((h, w))  # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P


img = cv2.imread("images/example.png")
img = cv2.resize(img, [150, 150])
Img_map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
Img_map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
grid = make_grid(img.shape[0], img.shape[1])
p = [[30, 30], [30, 75], [30, 120], [75, 30], [75, 75], [75, 120], [120, 30], [120, 75], [120, 120]]
q = [[35, 40], [23, 66], [40, 110], [75, 24], [75, 80], [75, 108], [120, 31], [120, 60], [124, 120]]
mls_affine_transform_pre_calc(p, grid, 1)
A_list, w, w_sum = mls_affine_transform_pre_calc(p, grid, 1)
new_v = (calculate_new_v(q, A_list, w, w_sum, grid))

k = 0
for j in enumerate(Img_map_x[:, 0]):
    for i in enumerate(Img_map_x[0, :]):
        Img_map_x[i[0], j[0]] = new_v[k][0]
        Img_map_y[i[0], j[0]] = new_v[k][1]
        k += 1
combined = np.stack((Img_map_x, Img_map_y), axis=2)

dst_img = cv2.remap(img, invert_map(combined)[:, :, 0].reshape(img.shape[0], img.shape[1]),
                    invert_map(combined)[:, :, 1].reshape(img.shape[0], img.shape[1]), cv2.INTER_LINEAR)
for coord in enumerate(p):
    cv2.circle(img, p[coord[0]], 3, (255, 255, 0), cv2.FILLED)
    cv2.circle(dst_img, q[coord[0]], 3, (255, 0, 0), cv2.FILLED)
    cv2.circle(dst_img, p[coord[0]], 3, (255, 255, 0), cv2.FILLED)

cv2.imshow("Deformed", dst_img)
cv2.imshow("Original", img)
cv2.waitKey()
