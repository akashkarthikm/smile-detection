# import packages
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time


# construct agrument parser and parse the argument
ap = argparse.ArgumentParser()
"""
ap.add_argument("-c", "--cascade", required = True,
    help = "path to where the face cascade resides")
ap.add_argument("-m", "--model", required = True,
    help = "path to pre-trained smile detector CNN")
"""
ap.add_argument("-v", "--video",
    help = "path to the (optional) video file")
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(".\haar_cascade\haarcascade_frontalface_default.xml")
model = load_model(".\model")

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])


# keep looping
feedback_given = False  # Flag to check if feedback has already been given
start_time = None  # Variable to store the start time when feedback is given
feedback_message = ""
satisfied_count = 0  # Counter for satisfied users
smcount = 0

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame to a larger size, convert it to grayscale, then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width=800)  # Set the desired width
    if not args.get("video", False):
        frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    # detect faces in the input frame, then clone the frame so
    # that we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via CNN
        roi = gray[fY: fY + fH, fX: fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # display bounding box rectangle 
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)
        
        # Check if feedback hasn't been given
        if not feedback_given:
            # determine the probabilities of both "smiling" and "not smiling"
            # then set the label accordingly
            (notSmiling, smiling) = model.predict(roi)[0]
            if smiling > notSmiling:
                label = "Smiling"
                smcount +=1 
            else:
                label = "Not Smiling"

            if label == "Not Smiling":
                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
            # Check if the label is "Smiling"
            if label == "Smiling" and smcount >3:
                feedback_message = "Thank you for Smiling !! Your feedback is recorded."
                feedback_given = True  # Set the flag to True to indicate feedback has been given
                start_time = time.time()  # Record the start time when feedback is given
                satisfied_count += 1  # Increment satisfied user count

    if feedback_given:
        # Display "Smile Detected" in red font above the rectangular box for 2 seconds
        elapsed_time = time.time() - start_time
        if elapsed_time < 2:
            cv2.putText(frameClone, "Smile Detected", (fX, fY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the feedback message
        cv2.putText(frameClone, feedback_message, (50, frameClone.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the timer countdown in numerical form for the remaining time (up to 10 seconds)
        remaining_time = max(0, 10 - elapsed_time)
        timer_message = f"~ {int(remaining_time)} seconds"
        cv2.putText(frameClone, timer_message, (50, frameClone.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Check if 10 seconds have passed since feedback is given
        if elapsed_time >= 10:
            feedback_given = False  # Reset the flag to False after 10 seconds
            smcount = 0

    # Display the satisfied user count at the top left of the screen
    cv2.putText(frameClone, f"Satisfied users: {satisfied_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # show our detected faces along with smiling/not smiling labels
    cv2.imshow("Face", frameClone)

    # if 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

print("satisfied_count:", satisfied_count)

#save to file with time 
curr_time = time.strftime("%H:%M:%S",time.localtime())
file = open("results.txt", "a") 
str = repr(satisfied_count) 
file.write("user count at time " + curr_time + " => " + str + "\n") 
file.close()
