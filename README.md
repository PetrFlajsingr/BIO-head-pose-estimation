# BIO-head-pose-estimation

Project for biometric systems course at FIT VUT. There are 3 methods implemented:

* Using 3D model of a face and PnP
* Using first frame for initialisation and computing head pose estimation via movement changes
* Using geometry and relationships between anthropometric points

Usage:
```
main.py [-h] [-i {image,video,camera}] [-p PATH] [-m {0,1,2}]

Head pose estimation (yaw, pitch, roll)

optional arguments:
  -h, --help            show this help message and exit
  -i {image,video,camera}
                        type of input: image, video, camera
  -p PATH               path to input file
  -m {0,1,2}            method used to estimate head pose: 0 - 3D model, 1 -
                        tracking, 2 - geometry

```

![alt text](https://github.com/PetrFlajsingr/BIO-head-pose-estimation/raw/master/data/demo.png "Demo")
