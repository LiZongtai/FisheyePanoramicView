import Stitcher
import cv2

if __name__ == '__main__':
    img_front = cv2.imread('res/front_0.png')
    img_right = cv2.imread('res/right_0.png')
    img_left = cv2.imread('res/left_0.png')
    img_rear = cv2.imread('res/rear_0.png')

    camera_imgs = {'front': img_front,
                   'right': img_right,
                   'left': img_left,
                   'rear': img_rear}
    camera_configs = Stitcher.get_default_camera_configs()
    stitcher = Stitcher.Stitcher(camera_configs, camera_imgs)
    stitched_img = stitcher.run()
    cv2.imwrite('stitched_img.png', stitched_img)
