import cv2
import numpy as np
import mediapipe as mp
import math

# ---------- Helpers ----------
def rotationMatrixToEulerAngles(R):
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll)
    Returns in degrees.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0

    return np.degrees([pitch, yaw, roll])


def classify_looking(yaw, pitch, yaw_thresh=25, pitch_thresh=20):
    """
    Decide if user is looking at screen based on yaw/pitch.
    """
    if abs(yaw) <= yaw_thresh and abs(pitch) <= pitch_thresh:
        return "LOOKING"
    return "NOT_LOOKING"

def wrap_angle(a):
    # wrap to [-180, 180]
    while a > 180:
        a -= 360
    while a < -180:
        a += 360
    return a

# ---------- Main ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,   # helps stability around eyes
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

# MediaPipe landmark indices
# (chosen for stability and solvePnP convenience)
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

# 3D model points (generic face model in mm-ish units)
# These values are approximate but widely used.
model_points = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -63.6, -12.5),     # Chin
    (-43.3, 32.7, -26.0),    # Left eye outer corner
    (43.3, 32.7, -26.0),     # Right eye outer corner
    (-28.9, -28.9, -24.1),   # Mouth left corner
    (28.9, -28.9, -24.1),    # Mouth right corner
], dtype=np.float64)

# Smoothing settings
alpha = 0.2  # higher = less smoothing
smoothed_angles = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    status = "NO_FACE"
    pitch = yaw = roll = 0.0

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]

        # Build 2D image points from selected landmarks
        image_points = []
        for k in LANDMARK_IDS:
            idx = LANDMARK_IDS[k]
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            image_points.append((x, y))

        image_points = np.array(image_points, dtype=np.float64)

        # Camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rot_matrix, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = rotationMatrixToEulerAngles(rot_matrix)
            pitch = wrap_angle(pitch)
            yaw = wrap_angle(yaw)
            roll = wrap_angle(roll)

            if pitch < -90:
                pitch += 180
            elif pitch > 90:
                pitch -= 180

            # Smooth angles
            angles = np.array([pitch, yaw, roll], dtype=np.float64)
            if smoothed_angles is None:
                smoothed_angles = angles
            else:
                smoothed_angles = alpha * angles + (1 - alpha) * smoothed_angles

            pitch, yaw, roll = smoothed_angles.tolist()

            status = classify_looking(yaw, pitch)

            # Draw landmarks used
            for (x, y) in image_points.astype(int):
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Draw head pose direction from nose tip
            nose_2d = tuple(image_points[0].astype(int))
            nose_3d = np.array((0, 0, 100.0))  # forward vector

            # project a point forward to show direction
            nose_end_2d, _ = cv2.projectPoints(
                nose_3d.reshape(1, 3),
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )
            p2 = tuple(nose_end_2d[0][0].astype(int))
            cv2.line(frame, nose_2d, p2, (255, 0, 0), 2)

    # Display
    cv2.putText(frame, f"Status: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Head Pose Attention Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
