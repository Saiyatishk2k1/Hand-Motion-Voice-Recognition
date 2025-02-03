import cv2
import numpy as np
from collections import deque
import mediapipe as mp
import speech_recognition as sr
import threading

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a blank image for drawing purposes
paintWindow = np.zeros((600, 800, 3), dtype=np.uint8)
paintWindow.fill(255)

# Colors and their names
colors = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (0, 255, 255),
    "black": (0, 0, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42)
}
color_keys = list(colors.keys())
current_color_name = color_keys[0]

# Initialize points dequeues for each color
points = {color_name: deque(maxlen=512) for color_name in colors}

# Initialize Mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Flag to track if the index finger tracking is paused
paused = False

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Flag to enable or disable voice commands
voice_enabled = False

def listen_command():
    with sr.Microphone() as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            # Use 'en-IN' for Indian English
            command = recognizer.recognize_google(audio, language='en-IN').lower()
            print(f"Command received: {command}")
            return command
        except sr.UnknownValueError:
            print("Could not understand the command.")
            return ""
        except sr.RequestError:
            print("Could not request results.")
            return ""
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return ""

def handle_voice_commands():
    global paused, current_color_name, voice_enabled
    while voice_enabled:
        command = listen_command()
        if command:
            if 'pause' in command:
                paused = True
            elif 'resume' in command:
                paused = False
            elif 'save' in command:
                cv2.imwrite('paint_image.png', paintWindow)
                print("Paint window saved as 'paint_image.png'")
            elif 'quit' in command:
                break
            elif 'erase' in command:
                current_color_name = 'black'
                print(f"Selected color: {current_color_name} (eraser mode)")
            elif 'clear' in command:
                for color_pts in points.values():
                    color_pts.clear()
                paintWindow.fill(255)
                print("Screen cleared.")
            else:
                for color_name in colors.keys():
                    if color_name in command:
                        current_color_name = color_name
                        print(f"Selected color: {current_color_name}")

def start_voice_thread():
    global voice_thread
    if voice_enabled:
        voice_thread = threading.Thread(target=handle_voice_commands, daemon=True)
        voice_thread.start()

def draw_ui():
    # Clear existing UI elements
    paintWindow[0:600, 0:100] = 255  # Clear left color palette area
    paintWindow[0:50, 100:800] = 255  # Clear top menu area

    # Draw color palette on the left side of the paint screen
    cv2.putText(paintWindow, 'Colors:', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    y_offset = 70
    for idx, color_name in enumerate(color_keys):
        color = colors[color_name]
        cv2.circle(paintWindow, (50, y_offset + idx * 50), 20, color, -1)
        if color_name == current_color_name:
            cv2.rectangle(paintWindow, (30, y_offset + idx * 50 - 20), (70, y_offset + idx * 50 + 20), (0, 0, 0), 2)

    # Add menu options at the top
    cv2.putText(paintWindow, 'Eraser: E', (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(paintWindow, 'Clear: C', (220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(paintWindow, 'Voice ON/OFF: V', (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(frame_rgb)

    # Process hand landmarks if tracking is not paused
    if not paused and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if idx == 8:
                    # Draw a circle at the tip of the index finger
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
                    # Append the coordinates to the deque of the selected color
                    points[current_color_name].append((cx, cy))
    elif paused:
        # Clear the current color points deque
        points[current_color_name].clear()

    # Draw lines on the paint window
    for color_name, color_pts in points.items():
        for i in range(1, len(color_pts)):
            if color_pts[i - 1] is None or color_pts[i] is None:
                continue
            cv2.line(paintWindow, color_pts[i - 1], color_pts[i], colors[color_name], 2)

    # Draw the UI elements
    draw_ui()

    # Show the frame and paint window
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    # Wait for key press
    key = cv2.waitKey(1)

    # Key actions
    if key == 27:  # Press 'Esc' to exit
        break
    elif key == 32:  # Press 'Spacebar' to pause/resume index finger tracking
        paused = not paused
        if paused:
            # Clear all color points dequeues when tracking is paused
            for color_pts in points.values():
                color_pts.clear()
    elif key == ord('e'):  # Press 'e' key to select the eraser
        current_color_name = 'black'  # Black color for erasing
    elif key == ord('c'):  # Press 'c' key to clear the paint screen
        for color_pts in points.values():
            color_pts.clear()
        paintWindow.fill(255)
    elif key == ord('s'):  # Press 's' key to save the paint screen
        cv2.imwrite('paint_image.png', paintWindow)
        print("Paint window saved as 'paint_image.png'")
    elif key == ord('q'):  # Press 'q' key to close the paint screen
        break
    elif key == ord('v'):  # Press 'v' key to toggle voice commands
        voice_enabled = not voice_enabled
        if voice_enabled:
            start_voice_thread()
        else:
            print("Voice commands disabled")
    else:
        # Map color key presses to their respective colors
        if 49 <= key <= 57:  # Keys '1' to '9'
            color_index = key - 49
            if color_index < len(color_keys):
                current_color_name = color_keys[color_index]
                print(f"Selected color: {current_color_name}")

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
