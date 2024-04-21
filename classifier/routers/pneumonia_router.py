import io
import tensorflow as tf

from fastapi import APIRouter, File
from PIL import Image
from keras.preprocessing.image import img_to_array

# Update the import path based on your project structure
from classifier.train_new import Train  # Assuming train.py is in the same directory

router = APIRouter()


@router.post('/predict')
def pneumonia_router(image_file: bytes = File(...)):
    """
    Predicts pneumonia from an uploaded image.

    Args:
        image_file (bytes): The image data in bytes format.

    Returns:
        dict: A dictionary containing the predicted class and pneumonia probability.
    """

    # Load the pre-trained model
    model = Train().define_model()
    model.load_weights('classifier/models/weights.h5')

    # Preprocess the image
    image = Image.open(io.BytesIO(image_file))

    # Handle color mode based on your training setup (RGB or grayscale)
    if image.mode != 'rgb' and image.mode != 'RGB':  # Check for both uppercase and lowercase
        image = image.convert('RGB')  # Assuming the model was trained on RGB images

    # Resize the image to match the training configuration
    image = image.resize((128, 128))  # Adjust based on your training setup (here 128x128)

    # Convert image to array and normalize
    image = img_to_array(image) / 255.0

    # Expand dimensions for compatibility with the model (add channel dimension)
    image = tf.expand_dims(image, axis=0)  # Assuming batch size of 1 for prediction

    # Make prediction
    prediction = model.predict(image)

    # Determine predicted class based on threshold (you can adjust the threshold)
    predicted_class = 'pneumonia' if prediction[0][0] > 0.5 else 'normal'

    return {'predicted_class': predicted_class, 'pneumonia_probability': str(prediction[0][0])}