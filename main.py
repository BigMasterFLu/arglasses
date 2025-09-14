import cv2
import mediapipe as mp
import numpy as np

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

    def create_keyboard_overlay(self, frame_shape, left_pos, right_pos):
        if self.keyboard_overlay is None or self.keyboard_overlay.shape != frame_shape:
            self.keyboard_overlay = np.zeros(frame_shape, dtype=np.uint8)
        
        # Clear the overlay
        self.keyboard_overlay.fill(0)
        
        # Create a semi-transparent blue keyboard
        keyboard_points = np.array([
            [left_pos[0], left_pos[1]],  # Top-left
            [right_pos[0], left_pos[1]],  # Top-right
            [right_pos[0], right_pos[1]],  # Bottom-right
            [left_pos[0], right_pos[1]]   # Bottom-left
        ], np.int32)
        
        # Draw filled polygon with blue tint
        cv2.fillPoly(self.keyboard_overlay, [keyboard_points], (150, 100, 0))  # BGR format
        
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
            left_hand_pos = None
            right_hand_pos = None
            
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
        
        return frame

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
            
            # If both hands are pinching, show the keyboard overlay
            if self.left_pinch and self.right_pinch and left_hand_pos and right_hand_pos:
                # Create the keyboard overlay
                overlay = self.create_keyboard_overlay(frame.shape, left_hand_pos, right_hand_pos)
                # Blend the overlay with the frame
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
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
                
            # Process the frame
            processed_frame = self.process_frame(frame)
            
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
