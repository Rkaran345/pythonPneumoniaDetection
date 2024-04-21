import os
import zipfile
import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Train:
    def __init__(self, force_download=False):
        self.force_download = force_download
        self.data_path = os.path.join(os.getcwd(), 'classifier/data/')
        self.model_path = os.path.join(os.getcwd(), 'classifier/models/')
        self.dataset_name = 'paultimothymooney/chest-xray-pneumonia'
        self.model: Sequential
        self.val: ImageDataGenerator
        self.train: ImageDataGenerator
        self.test: ImageDataGenerator
        self.epochs = 5

    def load_data(self):
        self.download_data()
        print("loading data")
        train_generator = ImageDataGenerator(rescale=1/255.0)
        test_generator = ImageDataGenerator(rescale=1/255.0)
        val_generator = ImageDataGenerator(rescale=1/255.0)
        self.train = train_generator.flow_from_directory(
            os.path.join(self.data_path, 'chest_xray/chest_xray/train'),
            target_size=(64, 64),
            batch_size=32,
            color_mode='rgb',  # Change color_mode to 'rgb'
            class_mode='binary'
        )
        self.test = test_generator.flow_from_directory(
            os.path.join(self.data_path, 'chest_xray/chest_xray/test'),
            target_size=(64, 64),
            batch_size=32,
            color_mode='rgb',  # Change color_mode to 'rgb'
            class_mode='binary'
        )
        self.val = val_generator.flow_from_directory(
            os.path.join(self.data_path, 'chest_xray/chest_xray/test'),
            target_size=(64, 64),
            batch_size=32,
            color_mode='rgb',  # Change color_mode to 'rgb'
            class_mode='binary'
        )
        logger.info('{} images in training set; {} are PNEUMONIA'.format(self.train.samples, sum(self.train.labels)))

    def download_data(self):
        print("downloading data")
        if self.force_download or (not os.path.isdir(self.data_path) and not os.path.isdir(self.models)):
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(self.dataset_name, path=self.data_path)
            with zipfile.ZipFile(os.path.join(self.data_path, 'chest-xray-pneumonia.zip')) as z:
                z.extractall(self.data_path)

    def define_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Freeze the pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        self.load_data()
        steps_per_epoch = self.train.samples // self.train.batch_size
        validation_steps = self.val.samples // self.val.batch_size
        self.model = self.define_model()
        self.model.fit(self.train,
                                 epochs=self.epochs,
                                 validation_data=self.val,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)

    def deploy_model(self):
        self.train_model()
        loss, acc = self.model.evaluate(self.test,
                                                  steps=self.test.samples // self.test.batch_size)
        logger.info('Model has been trained with loss, accuracy of {}, {}'.format(loss, acc))
        self.model.save_weights(os.path.join(self.model_path, 'weights.h5'))
        logger.info('Model weights have been saved to {}'.format(self.model_path))


if __name__ == '__main__':
    train_model = Train(force_download=False)
    train_model.deploy_model()
