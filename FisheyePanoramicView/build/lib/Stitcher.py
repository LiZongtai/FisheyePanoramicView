import json
import cv2
import numpy as np

default_camera_config_json = {
    "cameras": {
        "right": {
            "type": "fisheye",
            "width": 1280,
            "height": 720,
            "fx": 320.606,
            "fy": 320.311,
            "cx": 642.637,
            "cy": 348.676,
            "k1": 0.0752761,
            "k2": 0.0382620,
            "k3": -0.0224308,
            "k4": 0.00248036,
            "rms_in": 0.10602196,
            "rms_ex": 0.70746266,
            "rvec": [
                4.6590683600329474e-02,
                2.9007356060820682e+00,
                -1.2098754591223821e+00
            ],
            "tvec": [
                2.0464994767200611e+00,
                1.0830356652963020e+00,
                -1.9006220869678192e-01
            ]
        },
        "rear": {
            "type": "fisheye",
            "width": 1280,
            "height": 720,
            "fx": 3.1604215020904763e+02,
            "fy": 3.1604215020904763e+02,
            "cx": 6.5781462800653094e+02,
            "cy": 3.4484654372129120e+02,
            "k1": 7.8656168059824685e-02,
            "k2": 3.1612450312731022e-02,
            "k3": -1.9612367256880002e-02,
            "k4": 2.3425362039370423e-03,
            "rms_in": 1.0545598629783014e-01,
            "rms_ex": 1.0209525332593663e+00,
            "rvec": [
                1.4507759132997611e+00,
                1.4228700667371745e+00,
                -1.0575925411439542e+00
            ],
            "tvec": [
                8.5428122626631676e-03,
                8.6370578463909842e-01,
                -7.8713129147440608e-01
            ]
        },
        "left": {
            "type": "fisheye",
            "width": 1280,
            "height": 720,
            "fx": 319.390,
            "fy": 319.459,
            "cx": 641.912,
            "cy": 323.807,
            "k1": 0.0772143,
            "k2": 0.0373292,
            "k3": -0.0213480,
            "k4": 0.00228130,
            "rms_in": 0.1066287,
            "rms_ex": 0.8682844,
            "rvec": [
                2.3177724848377643e+00,
                -2.8233754591972367e-02,
                -7.2503823359083924e-02
            ],
            "tvec": [
                -1.9849147926266830e+00,
                1.2270790269920511e+00,
                -1.8340859525198064e-01
            ]
        },
        "front": {
            "type": "fisheye",
            "width": 1280,
            "height": 720,
            "fx": 319.753,
            "fy": 319.830,
            "cx": 663.327,
            "cy": 326.033,
            "k1": 0.0735571,
            "k2": 0.0451192,
            "k3": -0.0279534,
            "k4": 0.00396714,
            "rms_in": 0.098742,
            "rms_ex": 1.54295,
            "rvec": [
                1.5895022488286967e+00,
                -1.5854748434842987e+00,
                9.0834238700623571e-01
            ],
            "tvec": [
                -1.2115412288011036e-02,
                2.0893794128363705e+00,
                -3.0116512702224272e+00
            ]
        }
    }
}


def get_default_camera_configs():
    return default_camera_config_json['cameras']


def load_camera_config_from_json(camera_config_json):
    with open(camera_config_json) as load_f:
        content = load_f.read()
        load_f.close()
        a = json.loads(content)
        return a


def get_intrinsic_matrix(camera_config):
    fx = camera_config['fx']
    fy = camera_config['fy']
    cx = camera_config['cx']
    cy = camera_config['cy']
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def get_distortion_coefficients(camera_config):
    k1 = camera_config['k1']
    k2 = camera_config['k2']
    k3 = camera_config['k3']
    k4 = camera_config['k4']
    return np.array([k1, k2, k3, k4])


class Stitcher:
    Stitched_h = 512
    Stitched_w = 512
    car_h = 128
    car_w = 48

    cxy0 = 1
    cx0y = 2
    c0xy = 3
    cyx0 = 4
    cy0x = 5
    c0yx = 6

    def __init__(self, camera_configs, camera_imgs, measure=1, zero_axis=cxy0):
        self.camera_configs = camera_configs
        self.camera_imgs = camera_imgs
        self.measure = measure
        self.zero_axis = zero_axis

    def undistort(self, img, camera_config, scale=1):
        DIM = (camera_config['width'], camera_config['height'])
        K = get_intrinsic_matrix(camera_config)
        D = get_distortion_coefficients(camera_config)
        Knew = K.copy()
        if scale != 1:  # change fov
            Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]

        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def run(self):
        print("undistorrting...")
        undistorted_front = self.undistort(self.camera_imgs['front'], self.camera_configs['front'])
        undistorted_right = self.undistort(self.camera_imgs['right'], self.camera_configs['right'])
        undistorted_left = self.undistort(self.camera_imgs['left'], self.camera_configs['left'])
        undistorted_rear = self.undistort(self.camera_imgs['rear'], self.camera_configs['rear'])
        undistorted_imgs = {'front': undistorted_front,
                            'right': undistorted_right,
                            'left': undistorted_left,
                            'rear': undistorted_rear}
        stitched_img = self.stitch(undistorted_imgs)
        print("done!")
        return stitched_img

    def stitch(self, camera_imgs):
        print("stitching...")
        stitched_h = 512
        stitched_w = 512
        car_h = 128
        car_w = 48
        corner_w = 232
        corner_h = 192

        camera_img_front = camera_imgs['front']
        camera_img_right = camera_imgs['right']
        camera_img_left = camera_imgs['left']
        camera_img_rear = camera_imgs['rear']

        camera_config_front = self.camera_configs['front']
        camera_config_right = self.camera_configs['right']
        camera_config_left = self.camera_configs['left']
        camera_config_rear = self.camera_configs['rear']

        rvec_front = camera_config_front['rvec']
        tvec_front = camera_config_front['tvec']
        rvec_left = camera_config_left['rvec']
        tvec_left = camera_config_left['tvec']
        rvec_right = camera_config_right['rvec']
        tvec_right = camera_config_right['tvec']
        rvec_rear = camera_config_rear['rvec']
        tvec_rear = camera_config_rear['tvec']

        intrinsic_matrix_front = get_intrinsic_matrix(camera_config_front)
        distortion_coefficients_front = get_distortion_coefficients(camera_config_front)

        intrinsic_matrix_left = get_intrinsic_matrix(camera_config_left)
        distortion_coefficients_left = get_distortion_coefficients(camera_config_left)

        intrinsic_matrix_right = get_intrinsic_matrix(camera_config_right)
        distortion_coefficients_right = get_distortion_coefficients(camera_config_right)

        intrinsic_matrix_rear = get_intrinsic_matrix(camera_config_rear)
        distortion_coefficients_rear = get_distortion_coefficients(camera_config_rear)

        stitched_img = np.zeros([stitched_h, stitched_w, 3], np.uint8)
        # cv2.imshow("iamge", stitched_img)

        # front-left
        box_front_left = np.zeros([corner_h * corner_w, 3], np.float32)
        for i in range(0, corner_w):
            for j in range(0, corner_h):
                if self.zero_axis == self.cxy0:
                    box_front_left[i * corner_h + j, :] = np.float32([i * self.measure, j * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_front_left[i * corner_h + j, :] = np.float32([i * self.measure, 0, j * self.measure])
        box_front_left = box_front_left.reshape(-1, 3)
        p_front, Jacob = cv2.projectPoints(box_front_left, np.array(rvec_front), np.array(tvec_front),
                                           intrinsic_matrix_front, distortion_coefficients_front)
        p_left, Jacob = cv2.projectPoints(box_front_left, np.array(rvec_left), np.array(tvec_left),
                                          intrinsic_matrix_left, distortion_coefficients_left)

        for i in range(0, corner_w):
            for j in range(0, corner_h):
                p_u_front = int(p_front[i * corner_h + j][0][1])
                p_v_front = int(p_front[i * corner_h + j][0][0])
                p_u_left = int(p_left[i * corner_h + j][0][1])
                p_v_left = int(p_left[i * corner_h + j][0][0])

                if 0 < p_u_front < 1280 and 0 < p_v_front < 720:
                    stitched_img[j, i, :] = camera_img_front[p_v_front, p_u_front, :]
                if 0 < p_u_left < 1280 and 0 < p_v_left < 720:
                    stitched_img[j, i, :] = camera_img_left[p_v_left, p_u_left, :]

        # front
        box_front = np.zeros([car_w * corner_h, 3], np.float32)
        for i in range(0, car_w):
            for j in range(0, corner_h):
                if self.zero_axis == self.cxy0:
                    box_front[i * corner_h + j, :] = np.float32(
                        [(i + corner_w) * self.measure, j * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_front[i * corner_h + j, :] = np.float32(
                        [(i + corner_w) * self.measure, 0, j * self.measure])
        box_front = box_front.reshape(-1, 3)
        p_front, Jacob = cv2.projectPoints(box_front, np.array(rvec_front), np.array(tvec_front),
                                           intrinsic_matrix_front, distortion_coefficients_front)

        for i in range(0, car_w):
            for j in range(0, corner_h):
                p_u_front = int(p_front[i * corner_h + j][0][1])
                p_v_front = int(p_front[i * corner_h + j][0][0])
                if 0 < p_u_front < 1280 and 0 < p_v_front < 720:
                    stitched_img[j, i + corner_w, :] = camera_img_front[p_v_front, p_u_front, :]

        # front-right
        box_front_right = np.zeros([corner_h * corner_w, 3], np.float32)
        for i in range(0, corner_w):
            for j in range(0, corner_h):
                if self.zero_axis == self.cxy0:
                    box_front_right[i * corner_h + j, :] = np.float32(
                        [(i + corner_w + car_w) * self.measure, j * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_front_right[i * corner_h + j, :] = np.float32(
                        [(i + corner_w + car_w) * self.measure, 0, j * self.measure])
        box_front_right = box_front_right.reshape(-1, 3)
        p_front, Jacob = cv2.projectPoints(box_front_right, np.array(rvec_front), np.array(tvec_front),
                                           intrinsic_matrix_front, distortion_coefficients_front)
        p_right, Jacob = cv2.projectPoints(box_front_right, np.array(rvec_right), np.array(tvec_right),
                                           intrinsic_matrix_right, distortion_coefficients_right)

        for i in range(0, corner_w):
            for j in range(0, corner_h):
                p_u_front = int(p_front[i * corner_h + j][0][1])
                p_v_front = int(p_front[i * corner_h + j][0][0])
                p_u_right = int(p_right[i * corner_h + j][0][1])
                p_v_right = int(p_right[i * corner_h + j][0][0])

                if 0 < p_u_front < 1280 and 0 < p_v_front < 720:
                    stitched_img[j, i + corner_w + car_w, :] = camera_img_front[p_v_front, p_u_front, :]
                if 0 < p_u_right < 1280 and 0 < p_v_right < 720:
                    stitched_img[j, i + corner_w + car_w, :] = camera_img_right[p_v_right, p_u_right, :]

        # left
        box_left = np.zeros([corner_w * car_h, 3], np.float32)
        for i in range(0, corner_w):
            for j in range(0, car_h):
                if self.zero_axis == self.cxy0:
                    box_left[i * car_h + j, :] = np.float32(
                        [i * self.measure, (j + corner_h) * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_left[i * car_h + j, :] = np.float32(
                        [i * self.measure, 0, (j + corner_h) * self.measure])
        box_left = box_left.reshape(-1, 3)
        p_left, Jacob = cv2.projectPoints(box_left, np.array(rvec_left), np.array(tvec_left),
                                          intrinsic_matrix_left, distortion_coefficients_left)

        for i in range(0, corner_w):
            for j in range(0, car_h):
                p_u_left = int(p_left[i * car_h + j][0][1])
                p_v_left = int(p_left[i * car_h + j][0][0])
                if 0 < p_u_left < 1280 and 0 < p_v_left < 720:
                    stitched_img[j + corner_h, i, :] = camera_img_left[p_v_left, p_u_left, :]

        # right
        box_right = np.zeros([corner_w * car_h, 3], np.float32)
        for i in range(0, corner_w):
            for j in range(0, car_h):
                if self.zero_axis == self.cxy0:
                    box_right[i * car_h + j, :] = np.float32(
                        [(i + car_w) * self.measure, (j + corner_h) * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_right[i * car_h + j, :] = np.float32(
                        [(i + car_w) * self.measure, 0, (j + corner_h) * self.measure])
        box_right = box_right.reshape(-1, 3)
        p_right, Jacob = cv2.projectPoints(box_right, np.array(rvec_right), np.array(tvec_right),
                                           intrinsic_matrix_right, distortion_coefficients_right)

        for i in range(0, corner_w):
            for j in range(0, car_h):
                p_u_right = int(p_right[i * car_h + j][0][1])
                p_v_right = int(p_right[i * car_h + j][0][0])
                if 0 < p_u_right < 1280 and 0 < p_v_right < 720:
                    stitched_img[j + corner_h, i + car_w, :] = camera_img_right[p_v_right, p_u_right, :]

        # rear-left
        box_rear_left = np.zeros([corner_h * corner_w, 3], np.float32)
        for i in range(0, corner_w):
            for j in range(0, corner_h):
                if self.zero_axis == self.cxy0:
                    box_rear_left[i * corner_h + j, :] = np.float32(
                        [i * self.measure, (j + corner_h + car_h) * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_rear_left[i * corner_h + j, :] = np.float32(
                        [i * self.measure, 0, (j + corner_h + car_h) * self.measure])
        box_rear_left = box_rear_left.reshape(-1, 3)
        p_rear, Jacob = cv2.projectPoints(box_rear_left, np.array(rvec_rear), np.array(tvec_rear),
                                          intrinsic_matrix_rear, distortion_coefficients_rear)
        p_left, Jacob = cv2.projectPoints(box_rear_left, np.array(rvec_left), np.array(tvec_left),
                                          intrinsic_matrix_left, distortion_coefficients_left)

        for i in range(0, corner_w):
            for j in range(0, corner_h):
                p_u_rear = int(p_rear[i * corner_h + j][0][1])
                p_v_rear = int(p_rear[i * corner_h + j][0][0])
                p_u_left = int(p_left[i * corner_h + j][0][1])
                p_v_left = int(p_left[i * corner_h + j][0][0])

                if 0 < p_u_rear < 1280 and 0 < p_v_rear < 720:
                    stitched_img[(j + corner_h + car_h), i, :] = camera_img_rear[p_v_rear, p_u_rear, :]
                if 0 < p_u_left < 1280 and 0 < p_v_left < 720:
                    stitched_img[(j + corner_h + car_h), i, :] = camera_img_left[p_v_left, p_u_left, :]

        # rear
        box_rear = np.zeros([car_w * corner_h, 3], np.float32)
        for i in range(0, car_w):
            for j in range(0, corner_h):
                if self.zero_axis == self.cxy0:
                    box_rear[i * corner_h + j, :] = np.float32(
                        [(i + corner_w) * self.measure, (j + corner_h + car_h) * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_rear[i * corner_h + j, :] = np.float32(
                        [(i + corner_w) * self.measure, 0, (j + corner_h + car_h) * self.measure])
        box_rear = box_rear.reshape(-1, 3)
        p_rear, Jacob = cv2.projectPoints(box_rear, np.array(rvec_rear), np.array(tvec_rear),
                                          intrinsic_matrix_rear, distortion_coefficients_rear)

        for i in range(0, car_w):
            for j in range(0, corner_h):
                p_u_rear = int(p_rear[i * corner_h + j][0][1])
                p_v_rear = int(p_rear[i * corner_h + j][0][0])
                if 0 < p_u_rear < 1280 and 0 < p_v_rear < 720:
                    stitched_img[j + corner_h + car_h, i + corner_w, :] = camera_img_rear[p_v_rear, p_u_rear, :]

        # rear-right
        box_rear_right = np.zeros([corner_h * corner_w, 3], np.float32)
        for i in range(0, corner_w):
            for j in range(0, corner_h):
                if self.zero_axis == self.cxy0:
                    box_rear_right[i * corner_h + j, :] = np.float32(
                        [(i + corner_w + car_w) * self.measure, (j + corner_h + car_h) * self.measure, 0])
                elif self.zero_axis == self.cx0y:
                    box_rear_right[i * corner_h + j, :] = np.float32(
                        [(i + corner_w + car_w) * self.measure, 0, (j + corner_h + car_h) * self.measure])
        box_rear_right = box_rear_right.reshape(-1, 3)
        p_rear, Jacob = cv2.projectPoints(box_rear_right, np.array(rvec_rear), np.array(tvec_rear),
                                          intrinsic_matrix_rear, distortion_coefficients_rear)
        p_right, Jacob = cv2.projectPoints(box_rear_right, np.array(rvec_right), np.array(tvec_right),
                                           intrinsic_matrix_right, distortion_coefficients_right)

        for i in range(0, corner_w):
            for j in range(0, corner_h):
                p_u_rear = int(p_rear[i * corner_h + j][0][1])
                p_v_rear = int(p_rear[i * corner_h + j][0][0])
                p_u_right = int(p_right[i * corner_h + j][0][1])
                p_v_right = int(p_right[i * corner_h + j][0][0])

                if 0 < p_u_rear < 1280 and 0 < p_v_rear < 720:
                    stitched_img[(j + corner_h + car_h), i + corner_w + car_w, :] = camera_img_rear[p_v_rear, p_u_rear,
                                                                                    :]
                if 0 < p_u_right < 1280 and 0 < p_v_right < 720:
                    stitched_img[(j + corner_h + car_h), i + corner_w + car_w, :] = camera_img_right[p_v_right,
                                                                                    p_u_right, :]

        return stitched_img
