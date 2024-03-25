import dlib
import cv2

# Load the pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Load the image
image = cv2.imread("input_image.jpg")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray_image)

# Iterate over detected faces
for face in faces:
    # Predict facial landmarks
    landmarks = predictor(gray_image, face)

    # Iterate over facial landmarks
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        # Draw a landmark point on the image
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# Save the image with facial landmarks
cv2.imwrite("output_image_with_landmarks.jpg", image)

print("Image with facial landmarks saved successfully!")
