# Faical Tracking

Process and track human faces using OpenCV4 with a Raspberry Pi Camera mounted on a Pan/Tilt Platform.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Pan and Tilt Assembly:
https://www.robotshop.com/en/lynxmotion-large-pan-tilt-no-servos-black.html

RPI Physical Setup:
https://www.piddlerintheroot.com/16-channel-servo-controller/

OPENCV 4 RPI Installation:
https://www.pyimagesearch.com/2018/09/26/install-opencv-4-on-your-raspberry-pi/

### Installing

sudo pip install adafruit-pca9685

## Deployment

Initialize environment commands:
```
source ~/.profile
workon cv
sudo modprobe bcm2835-v4l2 
```

**Note** Create three directories in root folder prior to first run. "captures, dataset, and trainer"

Sequence to run python scripts:
1. registration.py - train a new face
2. sentry.py - detect/track faces

## Built With

* OpenCV4 - Open Source Computer Vision Framework
* Adafruit PCA9685 - Servo control library

## Authors

* **Andrew Johnson** - *Initial work*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

www.pyimagesearch.com
https://github.com/Mjrovai

