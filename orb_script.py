import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def orb(img1_input,img2_input):
    img1 = cv2.imread(img1_input)
    img2 = cv2.imread(img2_input)

    #creating the matrix
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance) 

    if len(kp1) == 0 or len(kp2) == 0:
        similarity_metric = 0
    else:
        similarity_metric = len(matches) / max(len(kp1), len(kp2))

    if similarity_metric > 0.19:
        st.write("These images are not similar")
    else:
        st.write("These images are the same")

    return ""

   

def plot_it(img1_input,img2_input):
    img1 = cv2.imread(img1_input)
    img2 = cv2.imread(img2_input)

    #creating the matrix
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
    
    st.image(matching_result)
    
    return 


# import streamlit as st

# def orb(img1_input, img2_input):
#     img1 = cv2.imread(img1_input)
#     img2 = cv2.imread(img2_input)

#     # Creating the matrix
#     orb = cv2.ORB_create()
#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)

#     # Brute Force Matching
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     matches = sorted(matches, key=lambda x: x.distance)

#     # Calculate similarity metric
#     if len(kp1) == 0 or len(kp2) == 0:
#         similarity_metric = 0
#     else:
#         similarity_metric = len(matches) / max(len(kp1), len(kp2))

#     if similarity_metric > 0.19:
#         st.write("False image")
#     else:
#         st.write("True")

#     return img1, kp1, des1, img2, kp2, des2, matches


# def plot_match_result(img1, kp1, des1, img2, kp2, des2, matches):
#     matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
#     st.image(matching_result, channels="BGR")


# if __name__ == "__main__":
#     img1_input = ""  # Add your image file path here
#     img2_input = ""  # Add your image file path here
#     img1, kp1, des1, img2, kp2, des2, matches = orb(img1_input, img2_input)
    
#     if st.button("Plot Match Result"):
#         plot_match_result(img1, kp1, des1, img2, kp2, des2, matches)


