from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/image classification/fabric.zip" -d "/content/data/"

import os

train_dir = "/content/data/fabric/train"
test_dir = "/content/data/fabric/test"

train_categories = [d for d in os.listdir(train_dir) if not d.startswith('.') ]
test_categories = [d for d in os.listdir(test_dir) if not d.startswith('.') ]

print("Train categories:", train_categories)
print("Test categories:", test_categories)

import tensorflow as tf

IMG_SIZE = (224,224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    labels='inferred'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    labels='inferred'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Class names:", class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))


material_mapping = {
    'canvas': 'Natural',
    'chambray': 'Natural',
    'chenille': 'Synthetic',
    'chiffon': 'Synthetic',
    'corduroy': 'Natural',
    'crepe': 'Natural',
    'denim': 'Natural',
    'faux_fur': 'Synthetic',
    'faux_leather': 'Synthetic',
    'flannel': 'Natural',
    'fleece': 'Synthetic',
    'gingham': 'Natural',
    'jersey': 'Synthetic',
    'knit': 'Synthetic',
    'lace': 'Natural',
    'lawn': 'Natural',
    'neoprene': 'Synthetic',
    'organza': 'Synthetic',
    'plush': 'Synthetic',
    'satin': 'Synthetic',
    'serge': 'Natural',
    'taffeta': 'Synthetic',
    'tulle': 'Synthetic',
    'tweed': 'Natural',
    'twill': 'Natural',
    'velvet': 'Synthetic',
    'vinyl': 'Synthetic'
}

material_class_mapping = {'Natural': 0, 'Synthetic': 1}

material_labels = [material_class_mapping[material_mapping[name]] for name in class_names]

def map_to_material_tensor(x, y):
    # y is an integer tensor
    material_label = tf.gather(material_labels, y)
    return x, material_label

train_material_ds = train_ds.map(map_to_material_tensor)
test_material_ds = test_ds.map(map_to_material_tensor)



import tensorflow as tf
from tensorflow.keras import layers, models

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

normalization_layer = layers.Rescaling(1./255)

num_material_classes = 2  # Natural / Synthetic

material_model = models.Sequential([
    layers.InputLayer(shape=(224, 224, 3)),

    # Apply augmentation first
    data_augmentation,

    # Normalize pixel values
    normalization_layer,

    # Conv Block 1
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Conv Block 2
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Conv Block 3
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    # Global Average Pooling
    layers.GlobalAveragePooling2D(),

    # Dense layer with Dropout
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),

    # Output layer
    layers.Dense(num_material_classes, activation='softmax')
])

material_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

material_model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

EPOCHS = 20

history = material_model.fit(
    train_material_ds,
    validation_data=test_material_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)
material_model.save('/content/drive/MyDrive/image classification/material_model.keras')
