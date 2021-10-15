# FisheyePanoramicView
Stitch 4 fisheye images to overlook panoramic view image base on reverse projection algorithm[1]
## How to use
```
import Stitcher
import cv2
...
camera_imgs = {'front': img_front,
               'right': img_right,
               'left': img_left,
               'rear': img_rear}
stitcher = Stitcher.Stitcher(camera_configs, camera_imgs)
stitched_img = stitcher.run()
...
```
ps:
1. load camera config json according to the format, use `get_default_camera_configs()` to get the default format
2. load camera images according to the format
## Cite
[1]赵三峰,谢明,陈玉明.基于逆向投影的全景泊车系统设计与实现[J].计算机工程与应用,2017,53(23):267-270.
