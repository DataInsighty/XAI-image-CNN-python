import torch
import matplotlib.pyplot as plt

def generate_saliency(model, input_image, device, target_class=None):
    """
    Function to generate a saliency map for a given model and input image.
    """
    input_image = input_image.unsqueeze(0).to(device)
    input_image.requires_grad_()

    # Forward pass
    output = model(input_image)
    
    # Use the top class if not specified
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Zero gradients
    model.zero_grad()

    # Backward pass
    output[:, target_class].backward()

    # Get the saliency map
    saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)
    
    # Create the figure for displaying saliency map
    fig, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(saliency[0].cpu().detach().numpy(), cmap='hot')

    # Add color bar to explain the color values
    cbar = plt.colorbar(img)
    cbar.set_label('Saliency Score', rotation=270, labelpad=20)

    ax.set_title(f"Saliency Map - Target Class {target_class}")
    ax.axis('off')

    return fig
