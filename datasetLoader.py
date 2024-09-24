# has the function to get images from the folder and prepare the dataset for training/testing

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_image_generators(train_dir, validation_dir, target_size=(32, 32), batch_size=32):

    # Training data generator with data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        rotation_range=20,  # Data augmentation parameters
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Validation data generator (only rescaling)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # Directory path
        target_size=target_size,  # Resize all images to this size
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator
