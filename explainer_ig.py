import torch
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

def plot_integrated_gradients(model, input_image, device, target_class=None):
    """
    Function to plot Integrated Gradients for a given model and input image.
    """
    model.eval()
    
    # Initialize IntegratedGradients
    ig = IntegratedGradients(model)

    # Prepare the input image and baseline
    input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension and move to device
    baseline = torch.zeros_like(input_image).to(device)  # Baseline is a black image

    # Get the model's prediction if target_class is not specified
    if target_class is None:
        with torch.no_grad():
            output = model(input_image)
            target_class = output.argmax(dim=1).item()  # Choose the class with highest probability

    # Compute attributions for the specified target class
    attributions = ig.attribute(input_image, baseline, target=target_class)
    
    # Convert to numpy for plotting
    attributions = attributions.squeeze(0).cpu().detach().numpy().sum(axis=0)
    
    # Create the figure for displaying attributions
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(attributions, cmap='hot')

    # Add a color bar to explain the color scale
    cbar = plt.colorbar(img)
    cbar.set_label('Attribution Score', rotation=270, labelpad=20)

    ax.set_title(f"Integrated Gradients - Target Class {target_class}")
    ax.axis('off')

    return fig
