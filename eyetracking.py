import numpy as np


class LandmarkProcessor:
    def __init__(self, landmarks, history_length=5):
        self.landmarks = landmarks
        self.HISTORY_LENGTH = history_length
        self.EAR_history = []

    def eye_aspect_ratio(self, eye_points):
        eye = np.array([(self.landmarks.part(n).x, self.landmarks.part(n).y) for n in eye_points])
        vertical_1 = np.linalg.norm(eye[1] - eye[5])
        vertical_2 = np.linalg.norm(eye[2] - eye[4])
        horizontal = np.linalg.norm(eye[0] - eye[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def calculate_ear(self):
        left_eye_points = range(36, 42)
        right_eye_points = range(42, 48)
        leftEAR = self.eye_aspect_ratio(left_eye_points)
        rightEAR = self.eye_aspect_ratio(right_eye_points)
        aveEAR = (leftEAR + rightEAR) / 2.0
        self.EAR_history.append(aveEAR)

        # Simply filter the EAR value to make it more smooth
        if len(self.EAR_history) > self.HISTORY_LENGTH:
            self.EAR_history.pop(0)

        smooth_ear = sum(self.EAR_history) / len(self.EAR_history)
        return smooth_ear