import streamlit as st
import torch
from PIL import Image
from explainer_gradcam import GradCAM, plot_grad_cam
from explainer_lime import lime_explainer
from explainer_ig import plot_integrated_gradients
from explainer_saliency import generate_saliency
from explainer_occlusion import plot_occlusion_sensitivity
from model import CNN_BT
import torchvision.transforms as transforms
import numpy as np

# Define the parameters for the CNN_BT model
params = {
    "shape_in": (3, 256, 256),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2
}

# Initialize the model and move it to the appropriate device (GPU/CPU)
model = CNN_BT(params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare the image for model input
def prepare_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Ensure image has 3 channels (RGB)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension: [1, C, H, W]
    return image_tensor.to(device)

# Predict function to pass image through model and get probabilities
def predict_fn(images, model, device):
    model.eval()
    tensor_images = torch.stack([transforms.ToTensor()(image) for image in images]).to(device)
    with torch.no_grad():
        outputs = model(tensor_images)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# Initialize session state for storing output images and label_index
if 'output_images' not in st.session_state:
    st.session_state['output_images'] = []
if 'label_index' not in st.session_state:
    st.session_state['label_index'] = None

# Center the logo using HTML and CSS
st.markdown(
    """
    <div style="text-align: center;">
        <img src="logo.jpg" width="300">
    </div>
    """, unsafe_allow_html=True
)

# App Title
st.markdown("<h1 style='text-align: center; color: black;'>Brain Tumor Classification with Explainability Techniques</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a brain scan image to predict if it's a brain tumor or healthy, and apply different explainability techniques to understand the model's decisions.</p>", unsafe_allow_html=True)

# Image uploader widget in a separate section
uploaded_file = st.file_uploader("Upload a brain scan image...", type=["jpg", "png", "jpeg", "tiff"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Brain Scan', use_column_width=True)

    # Prepare image for model input
    image_tensor = prepare_image(image)

    # Prediction Section
    if st.button('Predict'):
        label_index = predict_fn([image], model, device).argmax()
        label_name = {0: "Brain Tumor", 1: "Healthy"}[label_index]
        st.session_state['label_index'] = label_index
        st.write(f"**Predicted Label: {label_name}**")

    # Display rows for each explainability technique
    # Row 1: Grad-CAM
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Grad-CAM")
        st.write("Grad-CAM visualizes the regions in the image that are important for the model's decision.")
        if st.button("Generate Grad-CAM"):
            grad_cam = GradCAM(model, model.conv4, device)
            grad_cam_output = grad_cam(image_tensor)
            fig = plot_grad_cam(image_tensor.squeeze(0), grad_cam_output)
            st.session_state['output_images'].append(("Grad-CAM", fig))
    with col2:
        if st.session_state['output_images']:
            for title, fig in st.session_state['output_images']:
                if title == "Grad-CAM":
                    st.pyplot(fig)

    # Row 2: LIME Explanation
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("LIME Explanation")
        st.write("LIME highlights the parts of the image that influenced the model's decision.")
        if st.button("Generate LIME Explanation"):
            image_numpy = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            fig = lime_explainer(model, device, image_numpy)
            st.session_state['output_images'].append(("LIME", fig))
    with col2:
        if st.session_state['output_images']:
            for title, fig in st.session_state['output_images']:
                if title == "LIME":
                    st.pyplot(fig)

    # Row 3: Integrated Gradients
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Integrated Gradients")
        st.write("Integrated Gradients show the importance of each pixel by comparing it to a baseline.")
        if st.button("Generate Integrated Gradients"):
            fig = plot_integrated_gradients(model, image_tensor.squeeze(0), device)
            st.session_state['output_images'].append(("Integrated Gradients", fig))
    with col2:
        if st.session_state['output_images']:
            for title, fig in st.session_state['output_images']:
                if title == "Integrated Gradients":
                    st.pyplot(fig)

    # Row 4: Saliency Map
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Saliency Map")
        st.write("Saliency maps highlight the most important pixels for the model's prediction.")
        if st.button("Generate Saliency Map"):
            fig = generate_saliency(model, image_tensor.squeeze(0), device)
            st.session_state['output_images'].append(("Saliency Map", fig))
    with col2:
        if st.session_state['output_images']:
            for title, fig in st.session_state['output_images']:
                if title == "Saliency Map":
                    st.pyplot(fig)

    # Row 5: Occlusion Sensitivity
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Occlusion Sensitivity")
        st.write("Occlusion sensitivity identifies which parts of the image are most crucial for the model's decision.")
        if st.button("Generate Occlusion Sensitivity"):
            if st.session_state['label_index'] is not None:
                target_class = int(st.session_state['label_index'])
                fig = plot_occlusion_sensitivity(model, image_tensor.squeeze(0), device, target_class=target_class)
                st.session_state['output_images'].append(("Occlusion Sensitivity", fig))
            else:
                st.warning("Please run the prediction first to get the target class.")
    with col2:
        if st.session_state['output_images']:
            for title, fig in st.session_state['output_images']:
                if title == "Occlusion Sensitivity":
                    st.pyplot(fig)

    # Clear all outputs button at the end
    if st.button('Clear All'):
        st.session_state['output_images'] = []  # Clear all output images
        st.session_state['label_index'] = None  # Clear label index

else:
    st.write("Please upload an image first to start using the explainability techniques.")
