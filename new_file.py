import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
from io import BytesIOimport cv2
from orb.orb_script import orb
from orb.orb_script import plot_it





st.title("WELCOME TO YOUR IMAGE SIMILIARITY")
st.write("please enter first image")
uploaded_file1 = st.file_uploader("Upload image 1", type=['jpg', 'jpeg', 'png'])
if uploaded_file1 is not None:
    bytes_data1 = uploaded_file1.read()
    st.write("Filename:", uploaded_file1.name)
    image1 = Image.open(io.BytesIO(bytes_data1))
    f = plt.figure(figsize=(2, 3))
    plt.imshow(image1)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    st.pyplot(f)

st.write("please enter second image")
uploaded_file2 = st.file_uploader("Upload image 2", type=['jpg', 'jpeg', 'png'])
if uploaded_file2 is not None:
    bytes_data2 = uploaded_file2.read()
    st.write("Filename:", uploaded_file2.name)
    image2 = Image.open(io.BytesIO(bytes_data2))
    f = plt.figure(figsize=(2, 3))
    plt.imshow(image2)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    st.pyplot(f)





def faceVerification():
    if st.button('Verify Face'):
        
        img1_path = 'temp_img1.jpg'
        img2_path = 'temp_img2.jpg'
        with open(img1_path, 'wb') as f:
            f.write(uploaded_file1.getvalue())
        with open(img2_path, 'wb') as f:
            f.write(uploaded_file2.getvalue())
        result = orb(img1_path, img2_path)
        st.write(result)

        

faceVerification()

def plot():
     
    if st.button("Plot Match Result"):
            plot_it('temp_img1.jpg','temp_img2.jpg')

plot()

        
        




