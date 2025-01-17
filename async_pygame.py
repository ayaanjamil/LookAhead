import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np
import pygame
import time
from const import *
import eye_details
from sklearn.linear_model import SGDRegressor
import numpy as np

model_path = 'face_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

pygame.init()
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
pygame.display.set_caption("Face Landmarks with Mouse Position")

dot_radius = 20
dots = [
    {'pos': (100, 100), 'clicks': 0},  # Top-left
    {'pos': (screen_width // 2, 100), 'clicks': 0},  # Top-center
    {'pos': (screen_width - 100, 100), 'clicks': 0},  # Top-right
    {'pos': (100, screen_height // 2), 'clicks': 0},  # Middle-left
    {'pos': (screen_width // 2, screen_height // 2), 'clicks': 0},
    {'pos': (screen_width - 100, screen_height // 2), 'clicks': 0},  # Middle-right
    {'pos': (100, screen_height - 100), 'clicks': 0},  # Bottom-left
    {'pos': (screen_width // 2, screen_height - 100), 'clicks': 0},  # Bottom-center
    {'pos': (screen_width - 100, screen_height - 100), 'clicks': 0}  # Bottom-right
]

latest_result = None
cam = cv2.VideoCapture(0)

click_data = []
model_init = False

model_x = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01)
model_y = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.01)


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        for connection in mp.solutions.face_mesh.FACEMESH_TESSELATION:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = face_landmarks[start_idx]
            end_point = face_landmarks[end_idx]

            start_coord = (int(start_point.x * rgb_image.shape[1]),
                           int(start_point.y * rgb_image.shape[0]))
            end_coord = (int(end_point.x * rgb_image.shape[1]),
                         int(end_point.y * rgb_image.shape[0]))

            cv2.line(annotated_image, start_coord, end_coord, (0, 255, 0), 1)

        # Draw landmarks
        for landmark in face_landmarks:
            landmark_x = int(landmark.x * rgb_image.shape[1])
            landmark_y = int(landmark.y * rgb_image.shape[0])
            cv2.circle(annotated_image, (landmark_x, landmark_y), 1, (0, 0, 255), -1)

    return annotated_image


def face_landmark_callback(result: FaceLandmarkerResult, image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
    if result.face_landmarks:
        pass
        # print(eye_details.iris_position(result))


def on_mouse_click(mouse_pos):
    global model_init
    for dot in dots:
        dot_center = dot['pos']
        distance = np.sqrt((mouse_pos[0] - dot_center[0]) ** 2 + (mouse_pos[1] - dot_center[1]) ** 2)
        if distance <= dot_radius and latest_result:
            dot['clicks'] += 1
            # print(f"Dot at {dot_center} clicked {dot['clicks']} times.")
            break
    if latest_result:
        iris_coords = eye_details.iris_position(latest_result)

        scaled_iris_coords = []
        for coord in iris_coords:
            scaled_iris_coords.append([coord[0] * screen_width, coord[1] * screen_height])


        flattened_iris_coords = np.array(scaled_iris_coords).flatten().reshape(1, -1)
        model_x.partial_fit(flattened_iris_coords, np.array([mouse_pos[0]]))
        model_y.partial_fit(flattened_iris_coords, np.array([mouse_pos[1]]))
        model_init = True


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=face_landmark_callback,
    num_faces=1,
)

with FaceLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        timestamp_ms = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_result is not None and latest_result.face_landmarks:
            annotated_image = draw_landmarks_on_image(rgb_image, latest_result)
            display_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        else:
            display_frame = frame

        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = np.rot90(display_frame)
        frame_surface = pygame.surfarray.make_surface(display_frame)

        screen.fill((255, 255, 255))

        small_frame_surface = pygame.transform.scale(frame_surface, (320, 240))
        screen.blit(small_frame_surface, (0, 0))

        for dot in dots:
            dot_color = (0, 255, 0) if dot['clicks'] >= 5 else (255, 0, 0)  # Green if clicked 5+ times, else Red
            pygame.draw.circle(screen, dot_color, dot['pos'], dot_radius)

        # Real-time gaze prediction
        if model_init and latest_result is not None:
            iris_coords = eye_details.iris_position(latest_result)

            # Scale the iris coordinates
            scaled_iris_coords = []
            for coord in iris_coords:
                scaled_iris_coords.append([coord[0] * screen_width, coord[1] * screen_height])

            # Flatten and reshape the coordinates for prediction
            flattened_iris_coords = np.array(scaled_iris_coords).flatten().reshape(1, -1)

            predicted_click_x = model_x.predict(flattened_iris_coords)[0]
            predicted_click_y = model_y.predict(flattened_iris_coords)[0]

            # Clip predicted values to screen bounds
            predicted_click_x = int(min(max(predicted_click_x, 0), screen_width))
            predicted_click_y = int(min(max(predicted_click_y, 0), screen_height))

            # Draw the blue circle at predicted position
            pygame.draw.circle(screen, (0, 0, 255), (predicted_click_x, predicted_click_y), dot_radius)

        # Display mouse position
        mouse_x, mouse_y = pygame.mouse.get_pos()
        font = pygame.font.SysFont(None, 36)
        mouse_pos_text = font.render(f"Mouse Position: {mouse_x}, {mouse_y}", True, (0, 0, 0))
        screen.blit(mouse_pos_text, (10, screen_height - 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                print(click_data)
                cam.release()
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_mouse_click((mouse_x, mouse_y))

    cam.release()
    cv2.destroyAllWindows()
    pygame.quit()