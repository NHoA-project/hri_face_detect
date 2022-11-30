import tensorflow as tf
import numpy as np
from retinaface import RetinaFace
from PIL import Image
from typing import List, Tuple
import math
import cv2
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("/home/yusepp/full_model_crop.hdf5")

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





def extract_face(img: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Detects and return an image crop of the faces.

    Args:
        img (np.ndarray): numpy array containing the source image.

    Returns:
        A tuple containing 2 lists: cropped aligned faces and bounding boxes.

    """
    try:
        faces  = RetinaFace.detect_faces(img)
    except:
        return None, None, None
    
    crops = []
    bboxes = []
    landmarks = []
    for face in faces:
        if faces[face]['score'] >= 0.9:
            
            bbox = faces[face]['facial_area']
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            crop = alignment_procedure(left_eye=faces[face]['landmarks']['left_eye'], right_eye=faces[face]['landmarks']['right_eye'],
                                    nose=faces[face]['landmarks']['nose'], img=crop)
            #crop = cv2.resize(crop, (224, 224), interpolation = cv2.INTER_AREA)
            #crop = fix_size(crop, (224, 224))
            bboxes.append(bbox)
            crops.append(crop)
            landmarks.append(faces[face]['landmarks'])

    return crops, bboxes, landmarks


def extract_face_2(img, face_detector = None, align = True):

    from retinaface import RetinaFace
    from retinaface.commons import postprocess

    #---------------------------------

    # The BGR2RGB conversion will be done in the preprocessing step of retinaface.
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR

    """
    face = None
    img_region = [0, 0, img.shape[0], img.shape[1]] #Really?

    faces = RetinaFace.extract_faces(img_rgb, model = face_detector, align = align)

    if len(faces) > 0:
        face = faces[0][:, :, ::-1]

    return face, img_region
    """

    #--------------------------

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


    
    