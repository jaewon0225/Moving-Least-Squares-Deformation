import cv2
import numpy as np


def mls_affine_transform_pre_calc(p_input, v_input, a):
    ##Calculate weights w
    # convert p to nparray
    p, v = np.array(p_input), np.array(v_input)
    extruded_p = np.repeat(p[:, :, np.newaxis], len(v_input), axis=2)  # [p,2,v]

    v = v.reshape(len(v_input), 2)
    v = v.transpose()  # [2,v]
    # calculate w
    w = 1 / ((np.sum((extruded_p - v).astype(np.float32) ** 2, axis=1)) ** a)  # [p,v]

    ##Calculate p_star and q_star
    w = w.reshape(len(p_input), 1, len(v_input))
    w_sum = np.sum(w, axis=0)
    w_sum.reshape(1, len(v_input))
    p_star = np.sum(extruded_p * w, axis=0) / w_sum
    p_hat = extruded_p - p_star
    A_list = np.zeros([len(v_input), len(p_input)])
    for i in enumerate(v.transpose()):
        first_term = v.transpose()[i[0]] - p_star[:, i[0]]
        first_term.reshape(1, 2)
        weights = w[:, :, i[0]].reshape(len(p_input))
        second_term = np.zeros([2, 2])
        for j in enumerate(p_hat[:, :, i[0]]):
            p_j = j[1].reshape(1, 2)
            second_term += np.matmul(p_j.transpose(), p_j) * weights[j[0]]
        second_term = np.linalg.inv(second_term)
        for k in enumerate(p_hat[:, :, i[0]]):
            weighted_p_k = np.transpose(k[1].reshape(1, 2) * weights[k[0]])

            A_j = np.matmul(np.matmul(first_term, second_term), weighted_p_k)
            A_list[i[0], k[0]] = A_j
    return A_list, w, w_sum


def calculate_new_v(q_input, A_list, w, w_sum, v_input):
    q, v = np.array(q_input), np.array(v_input)
    v = v.reshape(len(v_input), 2)
    v = v.transpose()  # [2,v]
    extruded_q = np.repeat(q[:, :, np.newaxis], len(v_input), axis=2)  # [p,2,v]
    q_star = np.sum(extruded_q * w, axis=0) / w_sum
    q_hat = extruded_q - q_star

    ## Calculation done for p, calculate new values of v
    new_v_list = []
    for i in enumerate(v.transpose()):
        A_vector = A_list[i[0], :].reshape(len(q_input), 1)
        new_v = np.sum(q_hat[:, :, i[0]] * A_vector, axis=0).reshape(1, 2) + q_star[:, i[0]].reshape(1, 2)
        new_v_list.append(new_v.reshape(2))
    return np.array(new_v_list)


def make_grid(xdim, ydim):
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


class deformer():
    def set_parameters(self, p, img, a=1):
        self.p, self.a = p, a
        self.v = make_grid(img.shape[0], img.shape[1])

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
        dst_img = cv2.remap(img, invert_map(combined)[:, :, 0].reshape(img.shape[0], img.shape[1]),
                            invert_map(combined)[:, :, 1].reshape(img.shape[0], img.shape[1]), cv2.INTER_LINEAR)
        return dst_img
