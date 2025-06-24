#sentiment of child face
# by abeer azmat

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os


# Create model directory if needed
if not os.path.exists('models'):
    os.makedirs('models')

# Initialize CNN
classifier = Sequential()

# Convolution + Pooling Layers
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten
classifier.add(Flatten())

# Dense Layers
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dense(units= 4, activation = 'softmax'))   # 4 emotions
 
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Load Dataset
training_set = train_datagen.flow_from_directory(
                                                'dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical'
)

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical'
)

#train the CNN
classifier.fit(          
                         training_set,
                         samples_per_epoch = 8000 // 32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000 // 32 
)

# Save the model
classifier.save("models/emotion_model.h5")
print("✅ Multi-class model saved as models/emotion_model.h5")
