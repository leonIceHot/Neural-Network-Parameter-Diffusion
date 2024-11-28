import matplotlib.pyplot as plt
import numpy as np
from captum.attr import LayerGradCam, LayerAttribution

# Set the model to evaluation mode
model.eval()

# Select a sample image from the test dataset
data_iter = iter(test_loader)
images, labels = data_iter.next()
input_image = images[0].unsqueeze(0)
input_label = labels[0].item()

# Initialize LayerGradCam for the last convolutional layer
layer_gc = LayerGradCam(model, model.layer4[1].conv2)  # Using the last convolutional layer of ResNet18

# Compute Grad-CAM attributions
attributions_gc = layer_gc.attribute(input_image, target=input_label)

# Upsample the attributions to the size of the input image
attributions_gc = LayerAttribution.interpolate(attributions_gc, input_image.shape[2:])

# Convert attributions to a numpy array for visualization
attributions_gc = attributions_gc.squeeze().cpu().detach().numpy()

# Visualize the Grad-CAM heatmap
plt.imshow(attributions_gc, cmap='jet')
plt.colorbar()
plt.title(f'Grad-CAM for label: {input_label}')
plt.show()
