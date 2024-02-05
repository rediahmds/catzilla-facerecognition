import cv2
from simple_facerec import SimpleFacerec
from frameresizer.frameresizer import resize_frame


def main():

        # Encode faces from a folder
        sfr = SimpleFacerec()
        sfr.load_encoding_images("images/")

        # Load Camera
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Catzilla Cam", cv2.WINDOW_NORMAL)

        # Zoom factor (adjust as needed)
        zoom_factor = 2.0

        while True:
            ret, frame = cap.read()

            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                # Calculate the new face region
                new_x1 = int(x1 - (zoom_factor - 1) * (x2 - x1) / 2)
                new_y1 = int(y1 - (zoom_factor - 1) * (y2 - y1) / 2)
                new_x2 = int(x2 + (zoom_factor - 1) * (x2 - x1) / 2)
                new_y2 = int(y2 + (zoom_factor - 1) * (y2 - y1) / 2)

                # Ensure the new region is within the frame boundaries
                new_x1 = max(0, new_x1)
                new_y1 = max(0, new_y1)
                new_x2 = min(frame.shape[1], new_x2)
                new_y2 = min(frame.shape[0], new_y2)

                # Draw rectangle and text on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 4)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)

                # Crop and display the zoomed-in face
                zoomed_face = frame[new_y1:new_y2, new_x1:new_x2]
                cv2.imshow('Zoomed Face', zoomed_face)

            # Resize and display the original frame
            resized_frame = resize_frame(frame, scale=2.0)
            cv2.resizeWindow("Catzilla Cam", 720, 720)
            cv2.imshow("Catzilla Cam", resized_frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
