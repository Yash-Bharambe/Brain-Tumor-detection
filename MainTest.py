import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10Epochs.h5.keras')


image_path = r'import cv2'
from keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('BrainTumor10Epochs.h5.keras')


image_path = r'import cv2'
from keras.models import load_model
from PIL import Image
import numpy as np


model = load_model('BrainTumor10Epochs.h5.keras')


image_path = r'C:\Users\PARAS\Downloads\archive (1)\no\no0.jpg'
image = cv2.imread(image_path)


if image is None:
    print("Error: Could not read the image.")
else:
    # Convert the image to an array and then to a PIL Image
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Display the image (optional)
    img.show()

    # Preprocess the image for the model (this step depends on your model's expected input)
    img_resized = img.resize((224, 224))  # Resize image to match model's expected input size
    img_array = np.array(img_resized)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension


    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}")

image = cv2.imread(image_path)


if image is None:
    print("Error: Could not read the image.")
else:
    
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    
    img.show()

    
    img_resized = img.resize((224, 224))  
    img_array = np.array(img_resized)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    
    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}")

image = cv2.imread(image_path)


if image is None:
    print("Error: Could not read the image.")
else:

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    img.show()


    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}")
