import cv2
from simple_facerec import SimpleFacerec
from frameresizer.frameresizer import resize_frame

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)
cv2.namedWindow("Catzilla Cam", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)

    resizedframe = resize_frame(frame, scale=10.0)
    cv2.imshow("Catzilla Cam", resizedframe)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()