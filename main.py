import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import os

# Set environment variables for better performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations that might cause issues

# Ensure PyAutoGUI fails safely
pyautogui.FAILSAFE = True

class FingerTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        
        # Configure MediaPipe with optimized settings for maximum performance
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,  # Lower for faster detection
            min_tracking_confidence=0.6,   # Lower for faster tracking
            model_complexity=0  # Fastest model (0=lite, 1=full, 2=heavy)
        )
        
        print("🚀 MediaPipe configured for maximum performance (lite model)")
        
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
        self._showing_thumbs_up = False  # Track thumbs up state
        self._last_thumbs_up_time = 0  # Cooldown timer for thumbs up
        
        # Depth typing variables
        self.baseline_depth = {}  # Store baseline z-coordinate for each finger
        self.depth_threshold = 0.02  # How much closer finger needs to be to trigger
        self.last_key_press_time = {}  # Prevent rapid key presses
        self.key_press_cooldown = 0.5  # Cooldown between key presses
        self.is_pressing = False  # Track if currently in a press motion
        self.last_pressed_key = None  # Track last key to prevent repeats

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
        try:
            # Get relevant landmarks
            thumb_tip = hand_landmarks.landmark[4]
            thumb_mcp = hand_landmarks.landmark[2]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            # Check if thumb is pointing up (y position is significantly above MCP)
            thumb_up = thumb_mcp.y - thumb_tip.y > 0.08
            
            # Check if other fingers are folded (all below thumb tip)
            index_folded = index_tip.y > thumb_tip.y + 0.03
            middle_folded = middle_tip.y > thumb_tip.y + 0.03
            ring_folded = ring_tip.y > thumb_tip.y + 0.03
            pinky_folded = pinky_tip.y > thumb_tip.y + 0.03
            
            fingers_folded = index_folded and middle_folded and ring_folded and pinky_folded
            
            thumbs_up_detected = thumb_up and fingers_folded
            
            # Debug output
            if thumbs_up_detected:
                print("👍 Thumbs up detected!")
            
            return thumbs_up_detected
        except Exception as e:
            print(f"Error in check_thumbs_up: {e}")
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
        
    def check_depth_typing(self, hand_landmarks, h, w, is_right_hand):
        """Check if right index finger moved towards screen to type a key (single press)"""
        if not self.keyboard_visible or not is_right_hand:
            return
            
        current_time = time.time()
        index_tip = hand_landmarks.landmark[8]
        
        # Get finger position on screen
        finger_x = int(index_tip.x * w)
        finger_y = int(index_tip.y * h)
        finger_z = index_tip.z  # Depth coordinate (negative = closer to camera)
        
        # Initialize baseline depth if not set for right index finger
        if 'right_index' not in self.baseline_depth:
            self.baseline_depth['right_index'] = finger_z
            return
            
        # Check if finger moved significantly towards screen
        depth_diff = self.baseline_depth['right_index'] - finger_z
        
        if depth_diff > self.depth_threshold:
            # We're in the "pressed" zone
            if not self.is_pressing:
                # This is a new press motion
                self.is_pressing = True
                
                # Check which key the finger is over
                key = self.check_key_press((finger_x, finger_y))
                
                if key and key != self.last_pressed_key:
                    if key not in self.last_key_press_time:
                        self.last_key_press_time[key] = 0
                        
                    # Type the key if cooldown has passed
                    if (current_time - self.last_key_press_time[key]) > self.key_press_cooldown:
                        print(f"Right index depth typing: {key}")
                        # Handle special keys differently
                        if key == "space":
                            pyautogui.press("space")
                        elif key == "backspace":
                            pyautogui.press("backspace")
                        elif key == "enter":
                            pyautogui.press("enter")
                        else:
                            pyautogui.press(key.lower())
                        self.last_key_press_time[key] = current_time
                        self.last_pressed_key = key
        else:
            # We're back to normal depth - reset press state
            if self.is_pressing:
                self.is_pressing = False
                self.last_pressed_key = None
                print("🔄 Press motion completed - ready for next key")
                
        # Update baseline depth slowly to adapt to hand position
        self.baseline_depth['right_index'] = self.baseline_depth['right_index'] * 0.98 + finger_z * 0.02

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
        
        # Define keyboard layout with special keys
        keyboard_layout = [
            "1234567890",
            "QWERTYUIOP",
            "ASDFGHJKL;",
            "ZXCVBNM,./",
            "SPACE|BKSP|ENTER"  # Special keys row
        ]
        
        rows = len(keyboard_layout)
        
        # Calculate key dimensions for different rows
        standard_cols = len(keyboard_layout[0])  # Use first row as standard
        key_width = width // standard_cols
        key_height = height // rows
        
        # Draw horizontal lines
        for i in range(rows + 1):
            y = left_pos[1] + (i * key_height)
            cv2.line(self.keyboard_overlay, (left_pos[0], y), (right_pos[0], y), (200, 150, 50), 1)
        
        # Draw vertical lines for standard rows
        for i in range(standard_cols + 1):
            x = left_pos[0] + (i * key_width)
            cv2.line(self.keyboard_overlay, (x, left_pos[1]), (x, left_pos[1] + (rows - 1) * key_height), (200, 150, 50), 1)
        
        # Add letters and store key positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(key_width, key_height) * 0.025
        font_thickness = 1
        font_color = (255, 255, 255)  # White text
        
        for row in range(rows):
            if row < rows - 1:  # Standard rows (0-3)
                current_row = keyboard_layout[row]
                cols = len(current_row)
                
                for col in range(cols):
                    x = left_pos[0] + (col * key_width)
                    y = left_pos[1] + (row * key_height)
                    
                    # Store key position
                    key = current_row[col]
                    self.key_positions[key] = (x, y, key_width, key_height)
                    
                    # Draw key text
                    text_x = x + (key_width // 3)
                    text_y = y + (key_height * 2 // 3)
                    cv2.putText(self.keyboard_overlay, key, 
                              (text_x, text_y), font, font_scale, font_color, font_thickness)
            else:  # Special keys row (row 4)
                special_keys = keyboard_layout[row].split('|')
                special_key_width = width // len(special_keys)
                
                for i, special_key in enumerate(special_keys):
                    x = left_pos[0] + (i * special_key_width)
                    y = left_pos[1] + (row * key_height)
                    
                    # Draw vertical lines for special keys
                    if i < len(special_keys):
                        line_x = x + special_key_width
                        cv2.line(self.keyboard_overlay, (line_x, y), (line_x, y + key_height), (200, 150, 50), 1)
                    
                    # Map special key names to actual keys
                    if special_key == "SPACE":
                        actual_key = "space"
                        display_text = "SPACE"
                    elif special_key == "BKSP":
                        actual_key = "backspace"
                        display_text = "⌫"
                    elif special_key == "ENTER":
                        actual_key = "enter"
                        display_text = "↵"
                    else:
                        actual_key = special_key
                        display_text = special_key
                    
                    # Store key position
                    self.key_positions[actual_key] = (x, y, special_key_width, key_height)
                    
                    # Draw key text (centered for special keys)
                    text_size = cv2.getTextSize(display_text, font, font_scale, font_thickness)[0]
                    text_x = x + (special_key_width - text_size[0]) // 2
                    text_y = y + (key_height + text_size[1]) // 2
                    cv2.putText(self.keyboard_overlay, display_text, 
                              (text_x, text_y), font, font_scale, font_color, font_thickness)
        
        # Add lock status indicator
        lock_text = "LOCKED" if self.keyboard_locked else "UNLOCKED"
        lock_color = (0, 255, 0) if self.keyboard_locked else (0, 0, 255)  # Green if locked, red if unlocked
        cv2.putText(self.keyboard_overlay, lock_text,
                   (left_pos[0], left_pos[1] - 10), font, font_scale * 1.5, lock_color, 2)
        
        return self.keyboard_overlay



    def process_frame(self, frame):
        # Convert the BGR image to RGB for MediaPipe (no resizing for simplicity)
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
                
            # Check for thumbs up to lock/unlock keyboard with cooldown
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                current_time = time.time()
                if self.check_thumbs_up(results.multi_hand_landmarks[0]):
                    # Only toggle if enough time has passed since last toggle (1 second cooldown)
                    if current_time - self._last_thumbs_up_time > 1.0:
                        self.keyboard_locked = not self.keyboard_locked
                        self._last_thumbs_up_time = current_time
                        status = "LOCKED" if self.keyboard_locked else "UNLOCKED"
                        print(f"Keyboard {status}")
                        
                        if self.keyboard_overlay is not None and self.last_keyboard_points is not None:
                            # Get the correct points for keyboard recreation
                            left_pos = (self.last_keyboard_points[0][0], self.last_keyboard_points[0][1])
                            right_pos = (self.last_keyboard_points[2][0], self.last_keyboard_points[2][1])
                            
                            # Create keyboard with correct dimensions
                            self.keyboard_overlay = self.create_keyboard_overlay(
                                frame.shape,
                                left_pos,
                                right_pos
                            )
                    
            # Check for depth-based typing if keyboard is visible (right hand only)
            if self.keyboard_visible and not self.right_pinch and not self.left_pinch:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Determine if this is the right hand
                    is_right_hand = results.multi_handedness[idx].classification[0].label == "Right"
                    h, w, _ = frame.shape
                    # Use depth-based typing only for right hand
                    self.check_depth_typing(hand_landmarks, h, w, is_right_hand)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Optimize camera settings for performance
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        
        # Set the window size
        cv2.namedWindow('Finger Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Finger Tracking', 1300, 800)
        
        # Variables for FPS control and optimization
        fps_target = 60.0
        frame_time = 1.0 / fps_target
        skip_frame_count = 0
        frame_skip_interval = 1  # Process every frame for better responsiveness
        
        print("🎯 Camera optimized for high performance - targeting 60 FPS")
        
        while True:
            frame_start = cv2.getTickCount()
            
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Skip frame processing for performance if needed
            skip_frame_count += 1
            if skip_frame_count >= frame_skip_interval:
                skip_frame_count = 0
                # Process the frame
                processed_frame = self.process_frame(frame)
            else:
                processed_frame = frame
            
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
