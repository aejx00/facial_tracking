import os
import cv2
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

samples = 10 # number of facial samples taken for a new user


def get_user_id(my_path):
    """Parse files in /dataset for next available user id.
        Args:
            mypath: /dataset dir.
        Returns:
            integer user_id available for registration.
    """
    user_photos = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    if not user_photos:
        return 1
    user_numbers = list()
    for item in user_photos:
        user_numbers.append(int(item.split('.')[1]))
    new_id = max(user_numbers) + 1
    return new_id


def register(my_path, face_detector):
    """Ask user for name and save to file, after take face samples.
            Args:
                N/A
            Returns:
                N/A
    """

    face_id = get_user_id(my_path)
    user_name = input('\n Please enter your name and press <return>:  ')
    f = open("blacklist.txt", "a+")
    f.write(user_name+'\n')
    f.close()
    print("\n Initializing facial capture process. Look the camera and make different faces...")
    count = 0 # Initialize individual sampling face count
    cam = cv2.VideoCapture(0)  # init camera
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    while (True):
        ret, img = cam.read()
        img = cv2.flip(img, -1)  # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w]) # Save the captured image into the datasets folder
            print('User ' + str(face_id) + ' photo ' + str(count) + ' taken')
            cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'ESC' for exiting video
            cam.release()
            cv2.destroyAllWindows()
            break
        elif count >= samples:  # Take X number of face samples and stop video
            cam.release()
            cv2.destroyAllWindows()
            break


def getImagesAndLabels(my_path, face_detector):
    """Get all the images and label data accordingly.
            Args:
                path: /dataset dir.
            Returns:
                face samples and ids.
    """
    imagePaths = [os.path.join(my_path,f) for f in os.listdir(my_path)]
    faceSamples= list()
    ids = list()
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # convert to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids


if __name__ == '__main__':
    # register new user/get facial samples
    my_path = '/home/pi/Documents/FacialRecognitionProject/dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_detector = cv2.CascadeClassifier('/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    register(my_path, face_detector) # run registration process
    # create/update training data
    print("\n Training facial samples. Please Wait ...")
    faces, ids = getImagesAndLabels(my_path, face_detector)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')  # save model to trainer/trainer.yml
    print('\n Training data successfully generated')

