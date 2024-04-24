import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,  # Adjust as needed
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

def blur_faces(frame):
    """Detects faces and applies a blur to the facial mesh regions."""
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        # Create a blank mask for the mesh.
        mesh_mask = np.zeros((h, w), dtype=np.uint8)

        # Draw mesh lines based on facial landmarks.
        for face_landmarks in results.multi_face_landmarks:
            for connection in mp_face_mesh.FACEMESH_TESSELATION:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = face_landmarks.landmark[start_idx]
                end_point = face_landmarks.landmark[end_idx]
                start_point = (int(start_point.x * w), int(start_point.y * h))
                end_point = (int(end_point.x * w), int(end_point.y * h))
                cv2.line(mesh_mask, start_point, end_point, 255, thickness=20)  # Ensure correct thickness

        # Apply Gaussian blur to the mesh_mask to create a blurred effect.
        blurred_frame = cv2.GaussianBlur(frame_rgb, (99, 99), 30)  # Strong blur for facial areas

        # Apply the blurred mesh to the facial regions.
        blurred_faces = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mesh_mask)

        # Preserve non-blurred areas.
        non_blurred = cv2.bitwise_and(frame_rgb, frame_rgb, mask=cv2.bitwise_not(mesh_mask))

        # Combine blurred facial regions with the original frame.
        combined_frame_rgb = cv2.add(blurred_faces, non_blurred)

        # Convert the RGB array back to BGR for OpenCV.
        frame = cv2.cvtColor(combined_frame_rgb, cv2.COLOR_RGB2BGR)

    return frame
