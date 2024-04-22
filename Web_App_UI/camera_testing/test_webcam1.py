import cv2

def test_webcam(index):
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam {index}")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to retrieve frame from webcam {index}")
            break
        
        cv2.imshow(f"Webcam {index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

test_webcam(0)
