import cv2
import mediapipe as mp
import pyautogui
import time

# Setup
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Blink detection
blink_threshold = 0.004
last_click = 0
click_cooldown = 0.6  # seconds

# Main loop
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        # Eye tracking (using iris landmarks 474:478)
        for idx, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            if idx == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y

                # Clamp to safe zone to prevent fail-safe
                screen_x = max(10, min(screen_w - 10, screen_x))
                screen_y = max(10, min(screen_h - 10, screen_y))

                # Move mouse with small duration (faster)
                pyautogui.moveTo(screen_x, screen_y, duration=0.01)

        # Blink detection using landmarks [145, 159]
        top = landmarks[159].y
        bottom = landmarks[145].y
        blink_distance = top - bottom

        x1 = int(landmarks[145].x * frame_w)
        y1 = int(landmarks[145].y * frame_h)
        x2 = int(landmarks[159].x * frame_w)
        y2 = int(landmarks[159].y * frame_h)
        cv2.circle(frame, (x1, y1), 3, (0, 255, 255), -1)
        cv2.circle(frame, (x2, y2), 3, (0, 255, 255), -1)

        if blink_distance < blink_threshold and time.time() - last_click > click_cooldown:
            pyautogui.click()
            last_click = time.time()

    cv2.imshow('EyeNavShop - Fast Mode', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()
