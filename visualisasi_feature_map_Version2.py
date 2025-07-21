import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# 1. Load gambar dan pre-processing
img_path = 'foto_sample.jpg'  # Ganti dengan path gambar lokalmu
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 2. Load model CNN (pakai VGG16 pretrained, hanya layer awal)
model = VGG16(weights='imagenet', include_top=False)

# 3. Pilih layer convolutional yang ingin divisualisasikan
layer_name = 'block1_conv1'  # Coba layer awal (bisa diganti block2_conv1, dsb)
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# 4. Dapatkan feature map
feature_maps = intermediate_layer_model.predict(x)

# 5. Visualisasi beberapa feature map (misal 8 pertama)
num_filters = min(8, feature_maps.shape[-1])  # jumlah filter/feature map yang akan divisualisasikan
plt.figure(figsize=(15, 8))
for i in range(num_filters):
    plt.subplot(2, 4, i+1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')
    plt.title(f'Feature map {i+1}')
plt.suptitle(f'Feature maps from layer: {layer_name}')
plt.tight_layout()
plt.show()