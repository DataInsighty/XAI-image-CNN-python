import torch
import matplotlib.pyplot as plt
from captum.attr import Occlusion

def plot_occlusion_sensitivity(model, input_image, device, target_class=0):
    """
    Function to plot Occlusion Sensitivity for a given model and input image.
    """
    model.eval()
    occlusion = Occlusion(model)

    input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension if necessary

    # Sliding window occlusion
    attributions = occlusion.attribute(
        input_image,
        sliding_window_shapes=(3, 30, 30),  # Larger window size
        strides=(3, 30, 30),  # Increased stride
        target=target_class  # Specify target class for occlusion sensitivity
    )

    # Process attributions to be displayed
    attributions = attributions.squeeze(0).cpu().detach().numpy().sum(axis=0)

    # Create the figure for displaying the attributions
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(attributions, cmap='hot')
    ax.set_title(f"Occlusion Sensitivity for Class {target_class}")
    ax.axis('off')

    # Add color bar to explain the color scale
    cbar = plt.colorbar(img)
    cbar.set_label('Attribution Score', rotation=270, labelpad=15)

    return fig
