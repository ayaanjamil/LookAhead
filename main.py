import pyautogui
import cv2
import mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 386, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
font_thickness = 1


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    # frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]
    rgb_frame = mediapipe.Image(image_format=mediapipe.ImageFormat.SRGB, data=frame)

    detection_result = detector.detect(rgb_frame)

    if detection_result.face_landmarks:
        mesh_points = np.array(
            [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in detection_result.face_landmarks[0]])

        (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
        (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        cv2.circle(frame, center_left, int(l_radius), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, center_right, int(r_radius), (0, 0, 255), 1, cv2.LINE_AA)

        cv2.circle(frame, center_left, radius=2, color=(0, 0, 255), thickness=-1)
        cv2.circle(frame, center_right, radius=2, color=(255, 0, 0), thickness=-1)

        left_eye_bbox = cv2.boundingRect(mesh_points[LEFT_EYE])
        position_inside_left_eye = (
            (center_left[0] - left_eye_bbox[0]) / left_eye_bbox[2],
            (center_left[1] - left_eye_bbox[1]) / left_eye_bbox[3]
        )

        right_eye_bbox = cv2.boundingRect(mesh_points[RIGHT_EYE])
        position_inside_right_eye = (
            (center_right[0] - right_eye_bbox[0]) / right_eye_bbox[2],
            (center_right[1] - right_eye_bbox[1]) / right_eye_bbox[3]
        )
        """
        l_todisp = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, position_inside_left_eye))
        r_todisp = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, position_inside_right_eye))
        cv2.putText(frame, f"Left Iris Position: {l_todisp}", (10, 30), font, font_scale,
                    font_color, font_thickness)
        cv2.putText(frame, f"Right Iris Position: {r_todisp}", (10, 60), font, font_scale,
                    font_color, font_thickness)
        """

        blendshapes = detection_result.face_blendshapes
        # print(blendshapes)
        """
        highest_score = -1  # A low initial score
        highest_category_name = ''
        # Iterate through the categories to find the highest score
        for category in blendshapes[0]:
            if category.score > highest_score:
                highest_score = category.score
                highest_category_name = category.category_name
        print(highest_category_name)
        """

        if blendshapes[0][14].score > 0.50: # and blendshapes[0][15].score > 0.70
            pyautogui.moveRel(-30, 0)
        elif blendshapes[0][13].score > 0.50: # and blendshapes[0][16].score >0.70
            pyautogui.moveRel(30, 0)
        if blendshapes[0][17].score > 0.2 and blendshapes[0][18].score > 0.22:
            pyautogui.moveRel(0, -30)
        elif blendshapes[0][11].score > 0.40 and blendshapes[0][12].score > 0.40:
            pyautogui.moveRel(0, 30)
        if blendshapes[0][38].score > 0.96:
            pyautogui.click()


    cv2.imshow('img', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
