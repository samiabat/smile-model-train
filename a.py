import os
import cv2

# Load the pre-trained face detection cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Source folder containing the input images
source_folder = 'raw images/'

# Destination folder to store the extracted faces
destination_folder = 'extracted/'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate over image files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check if the file is an image
        # Load the image
        image_path = os.path.join(source_folder, filename)
        image = cv2.imread(image_path)

        # Convert the image to grayscale (required for face detection)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces and extract them
        for i, (x, y, w, h) in enumerate(faces):
            # Extract the face region
            face = image[y:y+h, x:x+w]

            # Save the extracted face to the destination folder
            destination_path = os.path.join(destination_folder, f'{filename.split(".")[0]}_face_{i}.jpg')
            cv2.imwrite(destination_path, face, [cv2.IMWRITE_JPEG_QUALITY, 100])

print("Faces extracted and saved successfully!")
