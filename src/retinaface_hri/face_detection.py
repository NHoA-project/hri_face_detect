import tensorflow as tf
import numpy as np
from retinaface import RetinaFace
from PIL import Image
from typing import List, Tuple
import math
import cv2
from keras.preprocessing import image
import mediapipe as mp

from retinaface import RetinaFace
from retinaface.commons import postprocess
from deepface import DeepFace

model = tf.keras.models.load_model("/home/yusepp/full_model_crop.hdf5")


from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def fix_size(img, target_size=(224, 224)):

    factor_0 = target_size[0] / img.shape[0]
    factor_1 = target_size[1] / img.shape[1]
    factor = min(factor_0, factor_1)
    
    dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
    img = cv2.resize(img, dsize)
    
    # Then pad the other side to the target size by adding black pixels
    diff_0 = target_size[0] - img.shape[0]
    diff_1 = target_size[1] - img.shape[1]
    
    img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
    
    
    img = cv2.resize(img, target_size)
    
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels /= 255
    
    return np.squeeze(img_pixels, axis=0)

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

#this function copied from the deepface repository: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
def alignment_procedure(img, left_eye, right_eye, nose):

    #this function aligns given face in img based on left and right eye coordinates

    #left eye is the eye appearing on the left (right eye of the person)
    #left top point is (0, 0)

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    #-----------------------
    #decide the image is inverse

    center_eyes = (int((left_eye_x + right_eye_x) / 2), int((left_eye_y + right_eye_y) / 2))
    
    if False:

        img = cv2.circle(img, (int(left_eye[0]), int(left_eye[1])), 2, (0, 255, 255), 2)
        img = cv2.circle(img, (int(right_eye[0]), int(right_eye[1])), 2, (255, 0, 0), 2)
        img = cv2.circle(img, center_eyes, 2, (0, 0, 255), 2)
        img = cv2.circle(img, (int(nose[0]), int(nose[1])), 2, (255, 255, 255), 2)

    #-----------------------
    #find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    #-----------------------

    #apply cosine rule

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        
        #PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0], floating point errors would produce cases violating this
        #In fact, we did come across a case where cos_a took the value 1.0000000169176173, which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))
        
        
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree

        #-----------------------
        #rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    #-----------------------

    return img #return img anyway

@timeit
def detect_smile(face: np.ndarray) -> bool:
    """This function detects if the provided image contains a smile.

    Args:
        face (np.ndarray): numpy array containing the cropped face image.

    Returns:
        Face Detection status. True for success, False otherwise.

    """
    score = model.predict(np.expand_dims(face, axis=0))[0][0]
    predicted = (score >= 0.5).astype('uint8')
    return predicted, score




@timeit
def extract_face(img, face_detector = None, align = True):

    obj = RetinaFace.detect_faces(img_path = img, model = face_detector, threshold = 0.9)
    crops = []
    bboxes = []
    landmarks_list = []
    
    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]


            detected_face = DeepFace.detectFace(img, target_size = (224, 224), detector_backend = 'retinaface')
            
            crops.append(detected_face)
            bboxes.append(facial_area)
            landmarks_list.append(identity["landmarks"])


    return crops, bboxes, landmarks_list




def extract_face_2(img, face_detector = None, align = True):

    from retinaface import RetinaFace
    from retinaface.commons import postprocess


    obj = RetinaFace.detect_faces(img_path = img, model = face_detector, threshold = 0.9)
    crops = []
    bboxes = []
    landmarks_list = []
    
    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]


            detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            if align:
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
           
                detected_face = postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)
                detected_face = fix_size(detected_face, (224, 224))
                
                crops.append(detected_face)
                bboxes.append(facial_area)
                landmarks_list.append(identity["landmarks"])


    return crops, bboxes, landmarks_list


@timeit
def get_nose_vector(face, results):
    face_2d = []
    face_3d = []
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = face.shape

    for idx, lm in enumerate(results):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            if idx == 1:
                nose_2d = (lm[0] * img_w, lm[1] * img_h)
                nose_3d = (lm[0] * img_w, lm[1] * img_h, lm[2] * 8000)

            x, y = int(lm[0] * img_w), int(lm[1] * img_h)

            # Get the 2D Coordinates
            face_2d.append([x, y])

            # Get the 3D Coordinates
            face_3d.append([x, y, lm[2]]) 
        
        
    # Convert it to the NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)

    # Convert it to the NumPy array
    face_3d = np.array(face_3d, dtype=np.float64) 


    # The camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])

    # The Distance Matrix
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    
    if y < -10:
        text = "Looking Left"
        text = 4
    elif y > 10:
        text = "Looking Right"
        text = 3
    elif x < -10:
        text = "Looking Down"
        text = 2
    else:
        text = "Forward"
        text = 1

    # Display the nose direction
    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

    return text, np.array(p1)-np.array(p2)


@timeit
def get_mesh(image):
    try:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process((image*255).astype(np.uint8))
                    
        for l in results.multi_face_landmarks:
            results = np.array([[landmark.x, landmark.y, landmark.z] for landmark in l.landmark])
        
        return results
    except:
        return []