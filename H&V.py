import cv2
import numpy as np
import mediapipe as mp
import speech_recognition as sr
import threading
from collections import deque

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a blank image for drawing
paintWindow = np.ones((600, 800, 3), dtype=np.uint8) * 255

# Colors and shortcuts
colors = {
    "red": (255, 0, 0),    # Red
    "green": (0, 255, 0),    # Green
    "blue": (0, 0, 255),    # Blue
    "yellow": (0, 255, 255),  # Yellow
    "purple": (128, 0, 128),  # Purple
    "orange": (255, 165, 0),  # Orange
    "black": (0, 0, 0)       # Black (eraser)
}
current_color = "red"
brush_size = 5
paused = False
voice_enabled = False

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.85)
mpDraw = mp.solutions.drawing_utils

# Track points for each color
points = {color: deque(maxlen=512) for color in colors}
recognizer = sr.Recognizer()

# Voice command function
def listen_command():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            print("Listening for command...")
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio, language='te-IN,en-IN').lower()
            print(f"Command received: {command}")
            return command
        except:
            return ""

def handle_voice_commands():
    global paused, current_color, brush_size, voice_enabled
    while voice_enabled:
        command = listen_command()
        if 'pause' in command or 'విరామం' in command:
            paused = True
        elif 'resume' in command or 'తిరిగి మొదలు పెట్టు' in command:
            paused = False
        elif 'erase' in command or 'తొలగించు' in command:
            current_color = 'black'
        elif 'clear' in command or 'అన్ని తుడిచివేయి' in command:
            for color_pts in points.values():
                color_pts.clear()
            paintWindow.fill(255)
        elif 'increase size' in command or 'పెంచు' in command:
            brush_size = min(20, brush_size + 2)
        elif 'decrease size' in command or 'తగ్గించు' in command:
            brush_size = max(2, brush_size - 2)
        else:
            for color in colors:
                if color in command:
                    current_color = color

def draw_ui():
    paintWindow[0:50, 0:800] = 255
    cv2.putText(paintWindow, f'Current Color: {current_color.upper()}  |  Brush Size: {brush_size}  |  Status: {"PAUSED" if paused else "ACTIVE"}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if not paused and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx == 8:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
                    points[current_color].append((cx, cy))
    
    if paused:
        for color in points:
            points[color].clear()
    
    for color, pts in points.items():
        for i in range(1, len(pts)):
            if pts[i - 1] and pts[i]:
                cv2.line(paintWindow, pts[i - 1], pts[i], colors[color], brush_size)
    
    draw_ui()
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    elif key == ord('c'):
        for pts in points.values():
            pts.clear()
        paintWindow.fill(255)
    elif key == ord('v'):
        voice_enabled = not voice_enabled
        if voice_enabled:
            threading.Thread(target=handle_voice_commands, daemon=True).start()
    elif key == ord('['):
        brush_size = max(2, brush_size - 2)
    elif key == ord(']'):
        brush_size = min(20, brush_size + 2)
    elif key == ord('p') or key == 32:  # 'p' or SPACEBAR to toggle pause
        paused = not paused

cap.release()
cv2.destroyAllWindows()
