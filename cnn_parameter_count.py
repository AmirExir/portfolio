"""
CNN Parameter Count Calculator

Create a basic CNN model in Keras with the following specifications:
- Input shape: (28, 28, 1)
- 2D Convolutional layer with 32 filters, kernel size of (3, 3), and 'relu' activation
- MaxPooling2D layer with pool size of (2, 2)
- Flatten layer to convert the 2D output into a 1D vector
- Dense layer with 128 units and 'relu' activation
- Output layer with 10 units and 'softmax' activation

Calculate the total number of parameters in the model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("="*70)
print("CNN MODEL ARCHITECTURE")
print("="*70)

# Create the model
model = keras.Sequential([
    # Input layer (implicitly defined by first layer)
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Display model summary
print("\nModel Summary:")
print("-"*70)
model.summary()

print("\n" + "="*70)
print("DETAILED PARAMETER CALCULATION")
print("="*70)

# Layer 1: Conv2D
conv_filters = 32
conv_kernel_size = (3, 3)
input_channels = 1
conv_params = (conv_kernel_size[0] * conv_kernel_size[1] * input_channels * conv_filters) + conv_filters
print(f"\n1. Conv2D Layer:")
print(f"   - Filters: {conv_filters}")
print(f"   - Kernel size: {conv_kernel_size}")
print(f"   - Input channels: {input_channels}")
print(f"   - Weights: {conv_kernel_size[0]} × {conv_kernel_size[1]} × {input_channels} × {conv_filters} = {conv_kernel_size[0] * conv_kernel_size[1] * input_channels * conv_filters}")
print(f"   - Biases: {conv_filters}")
print(f"   - Total parameters: {conv_params}")

# After Conv2D: output shape calculation
conv_output_h = 28 - 3 + 1  # (input - kernel + 2*padding) / stride + 1, with padding=0, stride=1
conv_output_w = 28 - 3 + 1
print(f"   - Output shape after Conv2D: ({conv_output_h}, {conv_output_w}, {conv_filters})")

# Layer 2: MaxPooling2D
pool_size = (2, 2)
pool_output_h = conv_output_h // pool_size[0]
pool_output_w = conv_output_w // pool_size[1]
print(f"\n2. MaxPooling2D Layer:")
print(f"   - Pool size: {pool_size}")
print(f"   - Parameters: 0 (pooling layers have no trainable parameters)")
print(f"   - Output shape after MaxPooling: ({pool_output_h}, {pool_output_w}, {conv_filters})")

# Layer 3: Flatten
flatten_size = pool_output_h * pool_output_w * conv_filters
print(f"\n3. Flatten Layer:")
print(f"   - Parameters: 0 (flatten has no trainable parameters)")
print(f"   - Output shape after Flatten: ({flatten_size},)")

# Layer 4: Dense (128 units)
dense1_units = 128
dense1_params = (flatten_size * dense1_units) + dense1_units
print(f"\n4. Dense Layer (128 units):")
print(f"   - Input size: {flatten_size}")
print(f"   - Output units: {dense1_units}")
print(f"   - Weights: {flatten_size} × {dense1_units} = {flatten_size * dense1_units}")
print(f"   - Biases: {dense1_units}")
print(f"   - Total parameters: {dense1_params}")

# Layer 5: Dense output (10 units)
dense2_units = 10
dense2_params = (dense1_units * dense2_units) + dense2_units
print(f"\n5. Output Dense Layer (10 units):")
print(f"   - Input size: {dense1_units}")
print(f"   - Output units: {dense2_units}")
print(f"   - Weights: {dense1_units} × {dense2_units} = {dense1_units * dense2_units}")
print(f"   - Biases: {dense2_units}")
print(f"   - Total parameters: {dense2_params}")

# Total parameters
total_params = conv_params + dense1_params + dense2_params

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nConv2D parameters:        {conv_params:>10,}")
print(f"MaxPooling2D parameters:  {0:>10,}")
print(f"Flatten parameters:       {0:>10,}")
print(f"Dense (128) parameters:   {dense1_params:>10,}")
print(f"Dense (10) parameters:    {dense2_params:>10,}")
print(f"{'-'*70}")
print(f"TOTAL PARAMETERS:         {total_params:>10,}")
print("="*70)

# Verify with Keras
print(f"\n✓ Verified with Keras model.count_params(): {model.count_params():,}")
print(f"✓ Match: {total_params == model.count_params()}")

print("\n" + "="*70)
print(f"ANSWER: {total_params:,} parameters")
print("="*70)
