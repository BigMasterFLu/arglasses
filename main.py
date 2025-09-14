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
            min_detection_confidence=0.2,  # Even lower threshold for better performance
            min_tracking_confidence=0.2,   # Lower tracking threshold
            model_complexity=0            # Use fastest model (0 is fastest, 1 is balanced, 2 is most accurate)
        )
        # FPS calculation variables
        self.fps_start_time = 0
        self.fps = 0
        self.frame_count = 0
        self.mp_draw = mp.solutions.drawing_utils
        # Key press tracking
        self.pressed_keys = {}  # Dictionary to store pressed keys and their timestamps
        self.key_press_delay = 1.0  # Delay in seconds between key presses
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
            
    def __init_thumbs_up_state(self):
        if not hasattr(self, '_last_thumbs_up_time'):
            self._last_thumbs_up_time = 0
        if not hasattr(self, '_thumbs_up_active'):
            self._thumbs_up_active = False
        if not hasattr(self, '_thumbs_up_start_time'):
            self._thumbs_up_start_time = 0
        if not hasattr(self, '_showing_thumbs_up'):
            self._showing_thumbs_up = False

    def check_thumbs_up(self, hand_landmarks):
        self.__init_thumbs_up_state()
        
        # Get relevant landmarks
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        wrist = hand_landmarks.landmark[0]
        
        # Get positions for all finger tips and joints
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        index_pip = hand_landmarks.landmark[6]
        middle_pip = hand_landmarks.landmark[10]
        ring_pip = hand_landmarks.landmark[14]
        pinky_pip = hand_landmarks.landmark[18]
        
        # Simplified thumb up check
        thumb_up = (thumb_mcp.y - thumb_tip.y > 0.08 and  # Reduced threshold for upward pointing
                   abs(thumb_tip.x - thumb_mcp.x) < 0.15)  # More tolerance for vertical alignment
        
        # Simplified fingers folded check
        fingers_folded = (
            index_tip.y > (index_pip.y - 0.02) and
            middle_tip.y > (middle_pip.y - 0.02) and
            ring_tip.y > (ring_pip.y - 0.02) and
            pinky_tip.y > (pinky_pip.y - 0.02)
        )
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Track thumbs up gesture duration
        if thumb_up and fingers_folded:
            if not self._showing_thumbs_up:
                self._thumbs_up_start_time = current_time
                self._showing_thumbs_up = True
                if current_time - self._last_thumbs_up_time > 0.3:  # Reduced delay
                    self._last_thumbs_up_time = current_time
                    return True
        else:
            if self._showing_thumbs_up:
                self._showing_thumbs_up = False
        
        return False
        
    def get_thumbs_up_duration(self):
        if self._showing_thumbs_up:
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            return current_time - self._thumbs_up_start_time
        return 0
        
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
        
        # Ensure valid dimensions
        width = right_pos[0] - left_pos[0]
        height = right_pos[1] - left_pos[1]
        if width < 50 or height < 50:  # Minimum size check
            return self.keyboard_overlay if self.keyboard_visible else None
        
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
                
                # Check if key was recently pressed
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                key_dark = False
                if key in self.pressed_keys:
                    time_since_press = current_time - self.pressed_keys[key]
                    if time_since_press < 1.0:  # Darken effect lasts for 1 second
                        key_dark = True
                        # Create darker background for pressed key
                        key_rect = np.array([
                            [x, y],
                            [x + key_width, y],
                            [x + key_width, y + key_height],
                            [x, y + key_height]
                        ], np.int32)
                        cv2.fillPoly(self.keyboard_overlay, [key_rect], (50, 30, 0))  # Darker blue
                    else:
                        # Remove old key presses
                        del self.pressed_keys[key]

                # Draw key text
                text_x = x + (key_width // 3)
                text_y = y + (key_height * 2 // 3)
                text_color = (150, 150, 150) if key_dark else font_color  # Dimmer text when key is dark
                cv2.putText(self.keyboard_overlay, key, 
                          (text_x, text_y), font, font_scale, text_color, font_thickness)
        
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
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # White dots
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)  # White lines
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
                # Validate hand positions
                if (abs(right_hand_pos[0] - left_hand_pos[0]) < frame.shape[1] and 
                    abs(right_hand_pos[1] - left_hand_pos[1]) < frame.shape[0] and
                    right_hand_pos[0] > left_hand_pos[0]):  # Ensure right hand is on the right
                    try:
                        self.keyboard_overlay = self.create_keyboard_overlay(frame.shape, left_hand_pos, right_hand_pos)
                        if self.keyboard_overlay is not None:
                            # Only update keyboard state if creation was successful
                            self.keyboard_visible = True
                            # Blend the overlay with the frame
                            frame = cv2.addWeighted(frame, 0.7, self.keyboard_overlay, 0.3, 0)
                    except Exception as e:
                        print(f"Error creating keyboard: {e}")
                        self.keyboard_visible = False
                        self.keyboard_overlay = None
                        self.last_keyboard_points = None
            # Keep the keyboard visible if it exists
            elif self.keyboard_visible and self.keyboard_overlay is not None and self.last_keyboard_points is not None:
                try:
                    # Use the last known good keyboard position
                    left_pos = (self.last_keyboard_points[0][0], self.last_keyboard_points[0][1])
                    right_pos = (self.last_keyboard_points[2][0], self.last_keyboard_points[2][1])
                    frame = cv2.addWeighted(frame, 0.7, self.keyboard_overlay, 0.3, 0)
                except Exception as e:
                    print(f"Error displaying keyboard: {e}")
                    # If there's an error, reset keyboard state
                    self.keyboard_visible = False
                    self.keyboard_overlay = None
                    self.last_keyboard_points = None
                
            # Check for thumbs up to lock/unlock keyboard - only check the first detected hand
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                if self.check_thumbs_up(results.multi_hand_landmarks[0]):
                    self.keyboard_locked = not self.keyboard_locked
                    if self.keyboard_overlay is not None and self.last_keyboard_points is not None:
                        # Get the correct points for keyboard recreation
                        left_pos = (self.last_keyboard_points[0][0], self.last_keyboard_points[0][1])
                        right_pos = (self.last_keyboard_points[2][0], self.last_keyboard_points[2][1])  # Use correct coordinates
                        
                        # Create keyboard with correct dimensions
                        self.keyboard_overlay = self.create_keyboard_overlay(
                            frame.shape,
                            left_pos,
                            right_pos
                        )
                    
            # Check for key presses if keyboard is visible and locked
            if self.keyboard_visible and self.keyboard_locked and not self.right_pinch and not self.left_pinch:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_tip = hand_landmarks.landmark[8]
                    h, w, _ = frame.shape
                    index_point = (int(index_tip.x * w), int(index_tip.y * h))
                    
                    # Check if index finger is touching a key
                    key = self.check_key_press(index_point)
                    if key:
                        current_time = cv2.getTickCount() / cv2.getTickFrequency()
                        # Check if enough time has passed since last press of this key
                        can_press = True
                        if key in self.pressed_keys:
                            time_since_press = current_time - self.pressed_keys[key]
                            if time_since_press < self.key_press_delay:
                                can_press = False
                        
                        if can_press:
                            # Store the press time
                            self.pressed_keys[key] = current_time
                            # Simulate key press
                            pyautogui.press(key.lower())
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for 60 FPS
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG for better performance
        
        # Set the window size
        cv2.namedWindow('Finger Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Finger Tracking', 2560, 1440)
        
        # Variables for FPS control
        fps_target = 60.0
        frame_time = 1.0 / fps_target
        
        while True:
            frame_start = cv2.getTickCount()
            
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
            
            # Calculate and display FPS
            self.frame_count += 1
            if self.frame_count >= 30:  # Update FPS every 30 frames
                current_time = cv2.getTickCount()
                if self.fps_start_time:
                    self.fps = 30.0 / ((current_time - self.fps_start_time) / cv2.getTickFrequency())
                self.fps_start_time = current_time
                self.frame_count = 0
            
            # Draw FPS counter
            cv2.putText(processed_frame, f'FPS: {self.fps:.1f}',
                       (processed_frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw thumbs up duration if active
            if self._showing_thumbs_up:
                duration = self.get_thumbs_up_duration()
                if duration >= 10:
                    message = "Thumbs up held for 10+ seconds!"
                else:
                    message = f"Thumbs up: {duration:.1f}s"
                cv2.putText(processed_frame, message,
                          (processed_frame.shape[1] - 400, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Finger Tracking', processed_frame)
            
            # Control frame rate
            frame_time_elapsed = (cv2.getTickCount() - frame_start) / cv2.getTickFrequency()
            wait_time = max(1, int((frame_time - frame_time_elapsed) * 1000))
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FingerTracker()
    tracker.run()
