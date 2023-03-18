import tensorflow as tf
import tensorflow.keras.layers as layers


class IdentityBlock:
    def __init__(self, filters, kernel_size, stride=1, padding='same'):
        super().__init__()
        self.conv1 = layers.Conv2D(filters[0], kernel_size,
                                   strides=stride, padding=padding)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters[1], kernel_size,
                                   strides=stride, padding=padding)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(filters[2], kernel_size,
                                   strides=stride, padding=padding)
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

    def __call__(self, x):
        FX = self.conv1(x)
        FX = self.bn1(FX)
        FX = self.relu1(FX)

        FX = self.conv2(FX)
        FX = self.bn2(FX)
        FX = self.relu2(FX)

        FX = self.conv3(FX)
        FX = self.bn3(FX)
        FX = self.relu3(FX)

        FX += x
        return FX


class ConvBlock:
    def __init__(self, filters, kernel_size, stride=1, padding='same'):
        super().__init__()
        self.conv1 = layers.Conv2D(filters[0], kernel_size,
                                   strides=stride, padding=padding)
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters[1], kernel_size,
                                   strides=stride, padding=padding)
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

        self.conv3 = layers.Conv2D(filters[2], kernel_size,
                                   strides=stride, padding=padding)
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.ReLU()

        self.conv4 = layers.Conv2D(filters[2], kernel_size,
                                   strides=stride, padding=padding)
        self.bn4 = layers.BatchNormalization()
        self.relu4 = layers.ReLU()

    def __call__(self, x):
        FX = self.conv1(x)
        FX = self.bn1(FX)
        FX = self.relu1(FX)

        FX = self.conv2(FX)
        FX = self.bn2(FX)
        FX = self.relu2(FX)

        FX = self.conv3(FX)
        FX = self.bn3(FX)
        FX = self.relu3(FX)

        FX += self.conv4(x)
        FX = self.bn4(FX)
        FX = self.relu4(FX)

        return FX


class ResNet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.pool1 = layers.MaxPool2D(3, strides=2, padding='same')

        self.conv2 = ConvBlock([64, 64, 256], 3)
        self.id1 = IdentityBlock([64, 64, 256], 3)
        self.id2 = IdentityBlock([64, 64, 256], 3)

        self.conv3 = ConvBlock([128, 128, 512], 3)
        self.id3 = IdentityBlock([128, 128, 512], 3)
        self.id4 = IdentityBlock([128, 128, 512], 3)
        self.id5 = IdentityBlock([128, 128, 512], 3)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def __call__(self, x):
        FX = self.conv1(x)
        FX = self.bn1(FX)
        FX = self.relu1(FX)
        FX = self.pool1(FX)

        FX = self.conv2(FX)
        FX = self.id1(FX)
        FX = self.id2(FX)

        FX = self.conv3(FX)
        FX = self.id3(FX)
        FX = self.id4(FX)
        FX = self.id5(FX)

        FX = self.avgpool(FX)
        FX = self.flatten(FX)
        FX = self.fc(FX)

        return FX