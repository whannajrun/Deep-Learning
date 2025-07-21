import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Load gambar dan preprocessing
img_path = 'foto_sample.jpg' # Ganti dengan path gambar lokal kamu!
img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # normalisasi
img_array = np.expand_dims(img_array, axis=0)  # tambahkan batch dimension

# Bangun model mini: Conv + ReLU + MaxPooling
model = Sequential([
    Conv2D(4, (3,3), activation='relu', input_shape=(128, 128, 1), name='conv1'),
    MaxPooling2D((2,2), name='pool1'),
])

# Dapatkan output convolution dan pooling
from tensorflow.keras import Model
conv_layer = model.get_layer('conv1')
pool_layer = model.get_layer('pool1')

conv_model = Model(inputs=model.input, outputs=conv_layer.output)
pool_model = Model(inputs=model.input, outputs=pool_layer.output)

conv_output = conv_model.predict(img_array)
pool_output = pool_model.predict(img_array)

# Visualisasi: Filter convolution dan hasil max pooling
plt.figure(figsize=(10,6))
for i in range(4):  # jumlah filter (4)
    # Convolution feature map
    plt.subplot(2, 4, i+1)
    plt.imshow(conv_output[0, :, :, i], cmap='gray')
    plt.title(f'Conv Filter {i+1}')
    plt.axis('off')
    # Max pooling output
    plt.subplot(2, 4, i+5)
    plt.imshow(pool_output[0, :, :, i], cmap='gray')
    plt.title(f'Pool {i+1}')
    plt.axis('off')
plt.suptitle('Visualisasi Feature Map:\nBaris 1 = Output Convolution | Baris 2 = Output Max Pooling', y=1.05)
plt.tight_layout()
plt.show()