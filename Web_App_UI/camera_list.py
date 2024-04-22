import cv2

def list_available_cameras():
    for i in range(10):  # Try up to 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print("Camera found at index", i)
            # Print out camera properties
            print("  Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            print("  Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print("  FPS:", cap.get(cv2.CAP_PROP_FPS))
            print("  FourCC:", int(cap.get(cv2.CAP_PROP_FOURCC)))
            cap.release()  # Release the capture object

if __name__ == "__main__":
    print("Listing available cameras:")
    list_available_cameras()
