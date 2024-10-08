import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activation = None

        # Hook for gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activation = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_image, class_idx=None):
        input_image = input_image.to(self.device)
        output = self.model(input_image)
        if class_idx is None:
            class_idx = output.argmax().item()

        # Zero the gradients and backward pass
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Get weights of the gradients and compute Grad-CAM
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        grad_cam = torch.sum(weights * self.activation, dim=1).squeeze(0).cpu().detach().numpy()

        # Apply ReLU and resize
        grad_cam = np.maximum(grad_cam, 0)
        grad_cam = cv2.resize(grad_cam, (input_image.shape[-1], input_image.shape[-2]))

        return grad_cam

def plot_grad_cam(image, grad_cam):
    fig, ax = plt.subplots()

    # Display the original image
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())

    # Overlay the Grad-CAM heatmap
    heatmap = ax.imshow(grad_cam, cmap='jet', alpha=0.5)

    # Create the colorbar linked to the heatmap
    fig.colorbar(heatmap, label="Importance")

    plt.title("Grad-CAM")
    plt.axis('off')
    
    return fig
