import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Ensure PyAutoGUI fails safely
pyautogui.FAILSAFE = True

class FingerTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_names = {
            4: "THUMB",
            8: "INDEX",
            12: "MIDDLE",
            16: "RING",
            20: "PINKY"
        }
        self.show_keyboard = False
        self.left_pinch = False
        self.right_pinch = False
        self.keyboard_overlay = None
        self.keyboard_visible = False
        self.last_keyboard_points = None
        self.keyboard_locked = False
        self.key_positions = {}  # Store key positions for hit detection

    def calculate_finger_angles(self, landmarks):
        angles = {}
        # Calculate angle for each finger (simplified)
        for finger_tip, name in self.finger_names.items():
            if finger_tip == 4:  # Thumb has different calculation
                continue
            
            tip = landmarks.landmark[finger_tip]
            pip = landmarks.landmark[finger_tip - 2]
            mcp = landmarks.landmark[finger_tip - 3]
            
            # Calculate 2D angle
            angle = np.degrees(np.arctan2(tip.y - pip.y, tip.x - pip.x) -
                             np.arctan2(mcp.y - pip.y, mcp.x - pip.x))
            angle = abs(angle)
            if angle > 180:
                angle = 360 - angle
            angles[name] = angle
        return angles

    def check_pinch(self, hand_landmarks, is_right_hand):
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calculate distance between thumb and index finger
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        # Update pinch state based on which hand
        if is_right_hand:
            self.right_pinch = distance < 0.05
        else:
            self.left_pinch = distance < 0.05
            
    def check_thumbs_up(self, hand_landmarks):
        # Get relevant landmarks
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        index_tip = hand_landmarks.landmark[8]
        
        # Check if thumb is pointing up (y position is significantly above MCP)
        thumb_up = thumb_mcp.y - thumb_tip.y > 0.15
        
        # Check if other fingers are folded (using index finger as example)
        other_fingers_folded = thumb_tip.y - index_tip.y < 0.1
        
        return thumb_up and other_fingers_folded
        
    def check_key_press(self, point):
        if not self.key_positions:
            return None
            
        for key, (x, y, w, h) in self.key_positions.items():
            if (point[0] >= x and point[0] <= x + w and 
                point[1] >= y and point[1] <= y + h):
                return key
        return None

    def create_keyboard_overlay(self, frame_shape, left_pos, right_pos):
        if self.keyboard_overlay is None or self.keyboard_overlay.shape != frame_shape:
            self.keyboard_overlay = np.zeros(frame_shape, dtype=np.uint8)
        
        # Clear the overlay and key positions
        self.keyboard_overlay.fill(0)
        self.key_positions.clear()
        
        # Create a semi-transparent blue keyboard
        keyboard_points = np.array([
            [left_pos[0], left_pos[1]],  # Top-left
            [right_pos[0], left_pos[1]],  # Top-right
            [right_pos[0], right_pos[1]],  # Bottom-right
            [left_pos[0], right_pos[1]]   # Bottom-left
        ], np.int32)
        
        # Save the keyboard points for persistence
        self.last_keyboard_points = keyboard_points
        
        # Draw filled polygon with blue tint
        cv2.fillPoly(self.keyboard_overlay, [keyboard_points], (150, 100, 0))  # BGR format
        
        # Calculate keyboard dimensions
        width = right_pos[0] - left_pos[0]
        height = right_pos[1] - left_pos[1]
        
        # Define keyboard layout
        keyboard_layout = [
            "1234567890",
            "QWERTYUIOP",
            "ASDFGHJKL;",
            "ZXCVBNM,./"
        ]
        
        rows = len(keyboard_layout)
        cols = len(keyboard_layout[0])
        key_width = width // cols
        key_height = height // rows
        
        # Draw horizontal lines
        for i in range(rows + 1):
            y = left_pos[1] + (i * key_height)
            cv2.line(self.keyboard_overlay, (left_pos[0], y), (right_pos[0], y), (200, 150, 50), 1)
        
        # Draw vertical lines and keys
        for i in range(cols + 1):
            x = left_pos[0] + (i * key_width)
            cv2.line(self.keyboard_overlay, (x, left_pos[1]), (x, right_pos[1]), (200, 150, 50), 1)
        
        # Add letters and store key positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(key_width, key_height) * 0.03  # Adjust font size based on key size
        font_thickness = 1
        font_color = (255, 255, 255)  # White text
        
        for row in range(rows):
            for col in range(cols):
                x = left_pos[0] + (col * key_width)
                y = left_pos[1] + (row * key_height)
                
                # Store key position
                key = keyboard_layout[row][col]
                self.key_positions[key] = (x, y, key_width, key_height)
                
                # Draw key text
                text_x = x + (key_width // 3)
                text_y = y + (key_height * 2 // 3)
                cv2.putText(self.keyboard_overlay, key, 
                          (text_x, text_y), font, font_scale, font_color, font_thickness)
        
        # Add lock status indicator
        lock_text = "LOCKED" if self.keyboard_locked else "UNLOCKED"
        lock_color = (0, 255, 0) if self.keyboard_locked else (0, 0, 255)  # Green if locked, red if unlocked
        cv2.putText(self.keyboard_overlay, lock_text,
                   (left_pos[0], left_pos[1] - 10), font, font_scale * 1.5, lock_color, 2)
        
        return self.keyboard_overlay



    def process_frame(self, frame):
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # Reset pinch states at the start of each frame
        self.left_pinch = False
        self.right_pinch = False
        left_hand_pos = None
        right_hand_pos = None
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine if this is the right or left hand
                is_right_hand = results.multi_handedness[idx].classification[0].label == "Right"
                
                # Draw the hand landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Get frame dimensions
                h, w, c = frame.shape
                
                # Check for pinch gesture
                self.check_pinch(hand_landmarks, is_right_hand)
                
                # Store hand positions for keyboard overlay
                index_tip = hand_landmarks.landmark[8]
                pos = (int(index_tip.x * w), int(index_tip.y * h))
                if is_right_hand:
                    right_hand_pos = pos
                else:
                    left_hand_pos = pos
                
                # Calculate angles for gesture recognition
                angles = self.calculate_finger_angles(hand_landmarks)
                
                # Display finger positions and information
                for finger_tip, name in self.finger_names.items():
                    # Get the landmark position
                    lm = hand_landmarks.landmark[finger_tip]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Draw circle at fingertip
                    cv2.circle(frame, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
                    
                    # Display finger name and position
                    text = f"{name}: ({cx}, {cy})"
                    cv2.putText(frame, text, (10, 30 + list(self.finger_names.values()).index(name) * 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display angle if available
                    if name in angles:
                        angle_text = f"Angle: {angles[name]:.1f}°"
                        cv2.putText(frame, angle_text, (200, 30 + list(self.finger_names.values()).index(name) * 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check for keyboard creation if not locked
            if not self.keyboard_locked and self.left_pinch and self.right_pinch and left_hand_pos and right_hand_pos:
                # Create the keyboard overlay
                self.keyboard_overlay = self.create_keyboard_overlay(frame.shape, left_hand_pos, right_hand_pos)
                # Blend the overlay with the frame
                frame = cv2.addWeighted(frame, 0.7, self.keyboard_overlay, 0.3, 0)
                self.keyboard_visible = True
            # Keep the keyboard visible if it exists
            elif self.keyboard_visible and self.keyboard_overlay is not None:
                frame = cv2.addWeighted(frame, 0.7, self.keyboard_overlay, 0.3, 0)
                
            # Check for thumbs up to lock/unlock keyboard
            for hand_landmarks in results.multi_hand_landmarks:
                if self.check_thumbs_up(hand_landmarks):
                    self.keyboard_locked = not self.keyboard_locked
                    if self.keyboard_overlay is not None:
                        # Refresh keyboard overlay to update lock status
                        self.keyboard_overlay = self.create_keyboard_overlay(
                            frame.shape, 
                            self.last_keyboard_points[0], 
                            self.last_keyboard_points[1]
                        )
                    break
                    
            # Check for key presses if keyboard is visible
            if self.keyboard_visible and not self.right_pinch and not self.left_pinch:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    h, w, _ = frame.shape
                    index_point = (int(index_tip.x * w), int(index_tip.y * h))
                    
                    # Check if index finger is touching a key
                    key = self.check_key_press(index_point)
                    if key:
                        # Simulate key press
                        pyautogui.press(key.lower())
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Set the window size
        cv2.namedWindow('Finger Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Finger Tracking', 1300, 800)
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # If keyboard was previously visible and we have saved points, show it
            if self.keyboard_visible and self.last_keyboard_points is not None:
                overlay = self.keyboard_overlay
                processed_frame = cv2.addWeighted(processed_frame, 0.7, overlay, 0.3, 0)
            
            # Display the frame
            cv2.imshow('Finger Tracking', processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FingerTracker()
    tracker.run()
