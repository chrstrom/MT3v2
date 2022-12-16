import numpy as np
import matplotlib.pyplot as plt

# Generate a random 10x10 matrix
matrix = np.random.rand(10, 10)

# Calculate the horizontal and vertical gradients
horizontal_gradient = np.gradient(matrix, axis=1)
vertical_gradient = np.gradient(matrix, axis=0)

# Plot all three matrices
fig, axs = plt.subplots(1, 3)

axs[0].imshow(matrix, cmap="gray")
axs[0].set_title("Original matrix")

axs[1].imshow(horizontal_gradient, cmap="gray")
axs[1].set_title("Horizontal gradient")

axs[2].imshow(vertical_gradient, cmap="gray")
axs[2].set_title("Vertical gradient")

plt.show()