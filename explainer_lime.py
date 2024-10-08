from lime import lime_image
import torch
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import matplotlib.patches as mpatches
import torchvision.transforms as transforms

# Function to pass image through the PyTorch model and get probabilities
def predict_fn(images, model, device):
    """
    Converts the images from NumPy to Torch tensors, passes them through the model, and returns probabilities.
    """
    model.eval()
    # Convert images from numpy to torch tensors
    tensor_images = torch.stack([transforms.ToTensor()(image) for image in images]).to(device)
    with torch.no_grad():
        outputs = model(tensor_images)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
    return probabilities

# LIME explainer function
def lime_explainer(model, device, input_image):
    """
    Generates a LIME explanation for the input image.
    """
    # Initialize LIME explainer for images
    explainer = lime_image.LimeImageExplainer()

    # Generate explanation for the input image
    explanation = explainer.explain_instance(
        input_image,  # Input image in NumPy format
        lambda imgs: predict_fn(imgs, model, device),  # Prediction function
        top_labels=1,  # Explain the top predicted class
        hide_color=0,  # Pixels to hide
        num_samples=1000  # Number of perturbations
    )

    # Get the explanation for the top class (most probable class)
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],  # Get explanation for top predicted label
        positive_only=False, 
        num_features=5,  # Number of superpixels to highlight
        hide_rest=False
    )

    # Plot the original image with LIME explanation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mark_boundaries(temp, mask))

    # Add labels for positive and negative contributions
    green_patch = mpatches.Patch(color='green', label='Positive contribution')
    red_patch = mpatches.Patch(color='red', label='Negative contribution')

    # Add the legend to explain the color coding
    ax.legend(handles=[green_patch, red_patch], loc='upper right')
    ax.set_title(f"LIME Explanation for Class {explanation.top_labels[0]}")
    ax.axis('off')

    return fig
