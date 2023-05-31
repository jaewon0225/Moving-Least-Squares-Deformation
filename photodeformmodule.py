import cv2
import numpy as np
from MLSDeformation import mls_affine_transform_pre_calc, calculate_new_v

class deformer():
    def make_grid(self, xdim, ydim):
        xgrid = np.linspace(0, xdim - 1, xdim)
        ygrid = np.linspace(0, ydim - 1, ydim)
        grid = []
        for i in xgrid:
            for j in ygrid:
                grid.append(np.array([i, j]))
        return np.array(grid)

    def invert_map(self, F):
        # shape is (h, w, 2), an "xymap"
        (h, w) = F.shape[:2]
        I = np.zeros_like(F)
        I[:, :, 1], I[:, :, 0] = np.indices((h, w))  # identity map
        P = np.copy(I)
        for i in range(10):
            correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
            P += correction * 0.5
        return P

    def set_parameters(self, p, img, a=1):
        self.p, self.a = p, a
        self.v = self.make_grid(img.shape[0], img.shape[1])

    def initialize_A(self):
        self.A_list, self.w, self.w_sum = mls_affine_transform_pre_calc(self.p, self.v, self.a)

    def deform_image(self, q, img):
        Img_map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        Img_map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        new_v = calculate_new_v(q, self.A_list, self.w, self.w_sum, self.v)
        k = 0
        for j in enumerate(Img_map_x[:, 0]):
            for i in enumerate(Img_map_x[0, :]):
                Img_map_x[i[0], j[0]] = new_v[k][0]
                Img_map_y[i[0], j[0]] = new_v[k][1]
                k += 1
        combined = np.stack((Img_map_x, Img_map_y), axis=2)
        dst_img = cv2.remap(img, self.invert_map(combined)[:, :, 0].reshape(img.shape[0], img.shape[1]),
                            self.invert_map(combined)[:, :, 1].reshape(img.shape[0], img.shape[1]), cv2.INTER_LINEAR)
        return dst_img