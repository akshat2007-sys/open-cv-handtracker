import cv2
import mediapipe as mp
import numpy as np
import time
import os


class AirWritingBoard:
    def __init__(self):
        self.model_path = "hand_landmarker.task"

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "hand_landmarker.task file not found. "
                "Please download it in the same folder as df.py"
            )

        self.cap = cv2.VideoCapture(0)

        self.canvas = None
        self.prev_x = None
        self.prev_y = None

        self.brush_size = 8
        self.eraser_size = 65
        self.brush_color = (255, 0, 255)

        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def get_pixel_position(self, landmark, width, height):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return x, y

    def fingers_up(self, landmarks):
        fingers = []

        # Index finger
        fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)

        # Middle finger
        fingers.append(1 if landmarks[12].y < landmarks[10].y else 0)

        # Ring finger
        fingers.append(1 if landmarks[16].y < landmarks[14].y else 0)

        # Pinky finger
        fingers.append(1 if landmarks[20].y < landmarks[18].y else 0)

        return fingers

    def draw_hand_landmarks(self, frame, landmarks, width, height):
        points = []

        for lm in landmarks:
            x, y = self.get_pixel_position(lm, width, height)
            points.append((x, y))
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for start, end in self.hand_connections:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    def write_with_finger(self, frame, index_x, index_y):
        cv2.circle(frame, (index_x, index_y), 12, self.brush_color, -1)

        if self.prev_x is None or self.prev_y is None:
            self.prev_x = index_x
            self.prev_y = index_y

        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (index_x, index_y),
            self.brush_color,
            self.brush_size
        )

        self.prev_x = index_x
        self.prev_y = index_y

    def erase_with_palm(self, frame, palm_x, palm_y):
        cv2.circle(frame, (palm_x, palm_y), self.eraser_size, (0, 0, 255), 3)

        cv2.circle(
            self.canvas,
            (palm_x, palm_y),
            self.eraser_size,
            (0, 0, 0),
            -1
        )

        self.prev_x = None
        self.prev_y = None

    def merge_canvas_with_frame(self, frame):
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        drawing_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        output = cv2.add(frame_bg, drawing_fg)
        return output

    def show_instructions(self, frame, mode):
        cv2.rectangle(frame, (0, 0), (900, 90), (0, 0, 0), -1)

        cv2.putText(
            frame,
            "Index Finger: Write | Open Palm: Erase | C: Clear | S: Save | Q: Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Mode: {mode}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    def run(self):
        with self.HandLandmarker.create_from_options(self.options) as landmarker:
            while True:
                success, frame = self.cap.read()

                if not success:
                    print("Camera not found")
                    break

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape

                if self.canvas is None:
                    self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                mode = "Idle"

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]

                    self.draw_hand_landmarks(frame, landmarks, width, height)

                    fingers = self.fingers_up(landmarks)

                    index_x, index_y = self.get_pixel_position(
                        landmarks[8],
                        width,
                        height
                    )

                    palm_x, palm_y = self.get_pixel_position(
                        landmarks[9],
                        width,
                        height
                    )

                    # Sirf index finger up = write
                    if fingers == [1, 0, 0, 0]:
                        mode = "Writing"
                        self.write_with_finger(frame, index_x, index_y)

                    # Open palm = erase
                    elif fingers == [1, 1, 1, 1]:
                        mode = "Erasing"
                        self.erase_with_palm(frame, palm_x, palm_y)

                    else:
                        mode = "Idle"
                        self.prev_x = None
                        self.prev_y = None

                else:
                    self.prev_x = None
                    self.prev_y = None

                output = self.merge_canvas_with_frame(frame)
                self.show_instructions(output, mode)

                cv2.imshow("Air Writing Board", output)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):
                    self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    self.prev_x = None
                    self.prev_y = None

                elif key == ord("s"):
                    filename = f"air_writing_{int(time.time())}.png"
                    cv2.imwrite(filename, self.canvas)
                    print(f"Saved as {filename}")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirWritingBoard()
    app.run()
