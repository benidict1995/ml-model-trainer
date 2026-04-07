
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2

# =====================
# CONFIGURATION
# =====================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Root dataset directory (must contain class subfolders)
DATASET_DIR = "dataset"

# Training epochs
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 5

# =====================
# LOAD DATASET
# =====================

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Extract class names automatically from folder names
class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)

# =====================
# SAVE LABELS
# =====================

# This file will be used later in Android / inference
with open("labels.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("Labels saved to labels.txt")

# =====================
# PREPROCESSING
# =====================

# Normalize pixel values to [0,1]
def preprocess(x, y):
    x = x / 255.0
    return x, y

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

# Optimize pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =====================
# DATA AUGMENTATION
# =====================

# Helps improve generalization
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# =====================
# BASE MODEL (TRANSFER LEARNING)
# =====================

# MobileNetV2 pretrained on ImageNet
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# =====================
# BUILD MODEL
# =====================

inputs = Input(shape=(224, 224, 3))

x = data_augmentation(inputs)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# =====================
# CALLBACKS
# =====================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =====================
# STAGE 1: TRAINING
# =====================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Stage 1: Initial Training")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=[early_stop]
)

# =====================
# STAGE 2: FINE-TUNING
# =====================

print("Stage 2: Fine-tuning")

# Unfreeze base model
base_model.trainable = True

# Freeze most layers, only fine-tune last layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-6),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[early_stop]
)

# =====================
# SAVE MODEL
# =====================

model.save("animal_model_v1.keras")

print("Model training complete and saved as animal_model_v1.keras")