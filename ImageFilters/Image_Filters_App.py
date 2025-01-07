import streamlit as st
import cv2
import numpy as np


st.title("Artistic Filters App")

uploaded_file = st.file_uploader("Upload an Image to add filters", ["JPG", "JPEG", "PNG"])

@st.cache_data
def bw_filter(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


@st.cache_data
def vignette(img, level=2):
    height, width = img.shape[:2]

    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)

    # Generating resultant_kernel matrix.
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    img_vignette = np.copy(img)

    # Apply the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask

    return img_vignette


@st.cache_data
def sepia(img):
    img_sepia = img.copy()
    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
    img_sepia = np.array(img_sepia, dtype=np.float64)
    img_sepia = cv2.transform(
        img_sepia, np.matrix([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])
    )
    # Clip values to the range [0, 255].
    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia, dtype=np.uint8)
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
    return img_sepia


@st.cache_data
def pencil_sketch(img, ksize=5):
    img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
    img_sketch, _ = cv2.pencilSketch(img_blur)
    return img_sketch


if uploaded_file is not None:
    # Convert the file to an opencv image.
    raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    input_col, output_col = st.columns(2)
    with input_col:
        st.header("Original")
        # Display uploaded image.
        st.image(img, channels="BGR", use_column_width=True)

    st.header("Filter Examples:")
    # Display a selection box for choosing the filter to apply.
    option = st.selectbox(
        "Select a filter:",
        (
            "None",
            "Black and White",
            "Sepia / Vintage",
            "Vignette Effect",
            "Pencil Sketch",
        ),
    )

    # Define columns for thumbnail images.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption("Black and White")
        st.image("filter_bw.jpg")
    with col2:
        st.caption("Sepia / Vintage")
        st.image("filter_sepia.jpg")
    with col3:
        st.caption("Vignette Effect")
        st.image("filter_vignette.jpg")
    with col4:
        st.caption("Pencil Sketch")
        st.image("filter_pencil_sketch.jpg")

    # Flag for showing output image.
    output_flag = 1
    # Colorspace of output image.
    color = "BGR"

    # Generate filtered image based on the selected option.
    if option == "None":
        # Don't show output image.
        output_flag = 0
    elif option == "Black and White":
        output = bw_filter(img)
        color = "GRAY"
    elif option == "Sepia / Vintage":
        output = sepia(img)
    elif option == "Vignette Effect":
        level = st.slider("level", 0, 5, 2)
        output = vignette(img, level)
    elif option == "Pencil Sketch":
        ksize = st.slider("Blur kernel size", 1, 11, 5, step=2)
        output = pencil_sketch(img, ksize)
        color = "GRAY"

    with output_col:
        if output_flag == 1:
            st.header("Output")
            st.image(output, channels=color)