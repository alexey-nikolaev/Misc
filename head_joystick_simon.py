from __future__ import division
import dlib
import cv2
import math
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from time import sleep

GAME_URL = 'https://www.openprocessing.org/sketch/678331'

PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

LANDMARKS = {
    'chin': 8,
    'nose_tip': 33,
    'left_eye_left_corner': 36,
    'right_eye_right_corner': 45,
    'mouth_left_corner': 48,
    'mount_right_corner': 54
}

def get_face_orientation(size, landmarks):
    
    image_points = np.array([
         landmarks['nose_tip'],
         landmarks['chin'],
         landmarks['left_eye_left_corner'],
         landmarks['right_eye_right_corner'],
         landmarks['mouth_left_corner'],
         landmarks['mount_right_corner']
     ], dtype="double")
    
    model_points = np.array([
      (0.0, 0.0, 0.0),             # Nose tip
      (0.0, -330.0, -65.0),        # Chin
      (-225.0, 170.0, -135.0),     # Left eye left corner (-165.0, 170.0, -135.0)
      (225.0, 170.0, -135.0),      # Right eye right corner (165.0, 170.0, -135.0)
      (-150.0, -150.0, -125.0),    # Left Mouth corner
      (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    #focal_length = size[1]
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double")
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
    
    axis = np.float32([[500,0,0], [0,500,0], [0,0,500]])
    
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
     
    pitch = math.degrees(math.asin(math.sin(pitch))) + 12.0
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw))) / 2.0

    return {'roll': roll, 'pitch': pitch, 'yaw': yaw}


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape[0], img.shape[1]
    
    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized


def detect_live():
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    camera = cv2.VideoCapture(0)
    
    driver = webdriver.Chrome()
    driver.get(GAME_URL)
    
    global previous_key_sent
    previous_key_sent = "0"
    
    global values_stack
    values_stack = {
        'pitch': [0 for i in range(20)],
        'roll': [0 for i in range(20)],
        'yaw': [0 for i in range(20)]
        }
    
    def send_keys(key):
        actions = ActionChains(driver)
        actions.send_keys(key)
        actions.perform()

    def send_key_check(key):
        global values_stack
        global previous_key_sent
        values_stack = {
            'pitch': [0 for i in range(20)],
            'roll': [0 for i in range(20)],
            'yaw': [0 for i in range(20)]
        }
        previous_key_sent = key
        send_keys(key)


    while True:
        
        ret, frame = camera.read()
        if ret == False:
            print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            break

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = resize(frame_grey, width=480)
        frame_flipped = cv2.flip(resize(frame, width=360), 1)

        dets = detector(frame_resized, 1)
        if len(dets) == 1:
            d = dets[0]
            shape = predictor(frame_resized, d)
            points = dict()
            for key in LANDMARKS:
                part = shape.part(LANDMARKS[key])
                points[key] = (int(part.x), int(part.y))
        
            face_orientation = get_face_orientation((frame_resized.shape[0], frame_resized.shape[1]), points)
            for dim in ['pitch', 'roll', 'yaw']:
                values_stack[dim].append(face_orientation[dim])
                values_stack[dim].pop(0)
            pitch, roll, yaw = np.median(values_stack['pitch']), np.median(values_stack['roll']), np.median(values_stack['yaw'])
            
            if roll > 15:
                send_key_check("2")
            elif yaw > 15:
                send_key_check("3")
            elif roll < -15:
                send_key_check("5")
            elif yaw < -15:
                send_key_check("6")
            elif abs(roll) < 5 and abs(yaw) < 5 and pitch < -7:
                send_key_check("4")
            elif abs(roll) < 5 and abs(yaw) < 5 and pitch > 7:
                send_key_check("1")
            else:
                if previous_key_sent != "0": send_key_check("0")
            info_string = ', '.join([name + ': ' + str(round(value, 2)) for name, value in zip(['pitch', 'roll', 'yaw'], [pitch, roll, yaw])])
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_flipped, info_string, (10,30), font, 0.5, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame_flipped, info_string, (10,30), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow("camera", frame_flipped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            driver.quit()
            break

if __name__ == "__main__":
    detect_live()
