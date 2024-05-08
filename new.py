from tensorflow.keras.models import Model,Sequential
#from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from joblib import load
import streamlit as st
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import io
from orb.orb_script import orb


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img= preprocess_input(img)
    return img

def findCosineSimiliarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c= np.sum(np.multiply(test_representation,test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def structural_sim(img1, img2):
    # Assuming the image is in 8-bit, ranges from 0 to 255
    sim, diff = ssim(img1, img2, data_range=img2.max() - img2.min(), full=True)
    return sim

# model = load('model_filename.joblib')
# vgg_face_descriptor= Model(inputs= model.layers[0].input, outputs=model.layers[-2].output)



epsilon = 0.44 #cosine similarity
#epsilon = 120 #euclidean distance
 
def verifyFace(img1, img2):
    model = load('model_filename.joblib')
    epsilon = 0.40 #cosine similarity
    vgg_face_descriptor= Model(inputs= model.layers[0].input, outputs=model.layers[-2].output)
    # img1_representation = vgg_face_descriptor.predict(preprocess_image('C:/Users/USER/Downloads/Database/%s' % (img1)))[0,:]
    # img2_representation = vgg_face_descriptor.predict(preprocess_image('C:/Users/USER/Downloads/Database/%s' % (img2)))[0,:]
    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
    
    cosine_similarity = findCosineSimiliarity(img1_representation, img2_representation)
    ssim_similarity = structural_sim(img1_representation, img2_representation)
    print(cosine_similarity)
    print(ssim_similarity)

    if(cosine_similarity < epsilon):
        st.write("verified... they are same person")
    else:
        st.write("unverified! they are not same person!")


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img= preprocess_input(img)
    return img

def preprocess_image(image_data):
    
    from PIL import Image
    import numpy as np
    
    # Load image data using PIL
    img = Image.open(io.BytesIO(image_data))
    
    # Resize the image
    img = img.resize((224, 224))
    
    # Convert image to array
    img = np.array(img)
    
    # Normalize pixel values between 0 and 1
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

# def verify(img1_path,img2_path):
#     # backends = ["opencv", "ssd", "mtcnn", "retinaface", "mediapipe","dlib"]
#     models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace","GhostFaceNet"]
    
#     img1 =preprocess_image(img1_path)
#     img2 =preprocess_image(img2_path)
#     result = DeepFace.verify(img1,img2,model_name = models[3],detector_backend= 'mtcnn')
#     return result

def verify(img1_path, img2_path):
    
    models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace","GhostFaceNet"]
    result = DeepFace.verify(img1_path,img2_path,model_name = models[3],detector_backend= 'mtcnn')
    
    if result["verified"] == False:
        st.write("The image is different")
    else:
        st.write("The image is the same")
    return " "
    
    

        