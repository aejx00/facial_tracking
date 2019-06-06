from __future__ import division
import os
import cv2
import time
import turret
import numpy as np
import RPi.GPIO as GPIO
import Adafruit_PCA9685

turret.gpio_setup()
#turret.cease_fire()

# configure servo and video settings
pan = 375
tilt = 375
frame_w = 640
frame_h = 480
pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)  # Set frequency to 60hz, good for servos.


def set_servo_pulse(channel, pulse):
    """Set servo pulse width.
        Args:
            channel: servo channel.
            pulse: pulse width value.
        Returns:
            N/A
    """
    pulse_length = 1000000  # 1,000,000 us per second
    pulse_length //= 60  # 60 Hz
    pulse_length //= 4096  # 12 bits of resolution
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)


def detect():
    """Detect faces of humans and translate face coordinates to new servo positions.
        Args:
            N/A
        Returns:
            N/A
    """
    targeted = False
    print("\n Initializing Sentry Application")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0 # id counter
    with open('blacklist.txt') as f: # load names in blacklist.txt into list
        names = f.read().splitlines()
    cam = cv2.VideoCapture(0) # initialize and start realtime video capture
    cam.set(3, frame_w)  # set video width
    cam.set(4, frame_h)  # set video height
    minW = 0.1 * cam.get(3) # define min window size to be recognized as a face
    minH = 0.1 * cam.get(4)
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, -1)  # flip vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )
        face_coord = {"x_axis": 0, "y_axis": 0}
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if targeted:
                cv2.circle(img, (x+120, y+90), 55, (0, 0, 255), 10)
            face_coord["x_axis"] = ((x + (x + w)) / 2) / 640
            face_coord["y_axis"] = ((y + (y + h)) / 2) / 480
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 100): # check if confidence is less them 100 ==> "0" is perfect match
                id = names[id]
                confidence_int = int(round(100 - confidence))
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "Unknown Entity"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
            #cv2.putText(img, str(face_coord["x_axis"]) + ',' + str(face_coord["y_axis"]), (x + 5, y * 2), font, 1,
            #            (0, 0, 255), 2)
        cv2.imshow('camera', img)
        # cv2.imwrite('/home/pi/Documents/FacialRecognitionProject/captures/' + str(int(time.time())) + '.jpg', img) # (optional) save snapshot of face
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pwm.set_pwm(0, 0, 375)
            pwm.set_pwm(1, 0, 375)
            cam.release()
            cv2.destroyAllWindows()
            break
        if ([i for i in faces]):
            face_center_x = faces[0, 0] + faces[0, 2] / 2
            face_center_y = faces[0, 1] + faces[0, 3] / 2
            err_x = 30 * (face_center_x - frame_w / 2) / (frame_w / 2)
            err_y = 30 * (face_center_y - frame_h / 2) / (frame_h / 2)
            if (err_x + 375) > 150 or (err_x + 375) < 600:
                pan = int(err_x + 375 + (0.7 * err_x))
                pwm.set_pwm(0, 0, pan)
                # print('pan: ' + str(pan))
            if (err_y + 375) > 150 or (err_y + 375) < 600:
                tilt = int(375 - (err_y + (0.7 * err_y)))
                pwm.set_pwm(1, 0, tilt)
                # print('tilt: ' + str(tilt))
            if pan > 600 or tilt > 600 or pan < 150 or tilt < 150:
                pan = 375
                tilt = 375
                pwm.set_pwm(1, 0, pan)
                pwm.set_pwm(0, 0, tilt)
            if confidence_int > 50:
                targeted = True
                turret.fire_semi_auto()
            else:
                targeted = False
    print("\n Terminating Application")


if __name__ == '__main__':
    try:
        detect()
    except Exception as e:
        print(e)
    finally:
        turret.cease_fire()
        GPIO.cleanup()
