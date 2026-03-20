import cv2
import os

person_name = input("Enter person name: ")

dataset_path = "../dataset/" + person_name

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

count = 0

while True:

    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (210,170,255), 3)

    # ---------- INFO PANEL ----------
    panel_x1, panel_y1 = 10, 10
    panel_x2, panel_y2 = 550, 150

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (240,235,255),
        -1
    )

    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ---------- TEXT ----------
    cv2.putText(
        frame,
        f"Person : {person_name}",
        (25,45),
        font,
        0.8,
        (50,40,90),
        2
    )

    cv2.putText(
        frame,
        f"Images Captured : {count}",
        (25,75),
        font,
        0.8,
        (50,40,90),
        2
    )

    cv2.putText(
        frame,
        "Press SPACE to Capture Image",
        (25,110),
        font,
        0.7,
        (60,50,100),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (25,135),
        font,
        0.7,
        (60,50,100),
        2
    )

    cv2.imshow("Face Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):

        if len(faces) > 0:
            x,y,w,h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100,100))

            file_name = dataset_path + f"/{count}.jpg"
            cv2.imwrite(file_name, face)

            count += 1

            print("Captured image:", count)

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()