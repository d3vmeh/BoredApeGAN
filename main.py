import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#import numpy as np
import matplotlib.pyplot as plt
#import os
from keras import losses
from keras import layers
import numpy as np
import os
from tqdm import tqdm


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0],True)
#from tqdm import tqdm
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


sizex = 128
sizey=128
dataset = keras.preprocessing.image_dataset_from_directory(
    directory="data",label_mode=None,image_size=(sizex,sizey),batch_size=32,shuffle=True
).map(lambda x: x/255.0)


discriminator = keras.Sequential([
    keras.Input(shape=(sizex,sizey,3)),
    layers.Conv2D(sizex,kernel_size=4,strides=2,padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2D(sizex*2,kernel_size=4,strides=2,padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2D(sizey*2,kernel_size=4,strides=2,padding="same"),
    layers.LeakyReLU(0.2),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(1,activation="sigmoid")
])

latent_dim = sizex*2

generator = keras.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(int(sizex/8)*int(sizex/8)*sizex*2),
    layers.Reshape((int(sizex/8),int(sizex/8),sizex*2)),   #creating an x/8 by y/8 image
    layers.Conv2DTranspose(sizex*2,kernel_size=4,strides=2,padding="same"), #doubles dimensions of the image
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(sizex*2,kernel_size=4,strides=2,padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(sizex*4,kernel_size=4,strides=2,padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2D(3,kernel_size=5,padding="same",activation="sigmoid")

])

generator.summary()

opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)
loss = losses.BinaryCrossentropy()


print('hello')
for epoch in range(1000):
    for index, real in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size,latent_dim))

        #with tf.GradientTape() as gen_tape:
        fake = generator(random_latent_vectors)


        if index%100 == 0:
            img = keras.preprocessing.image.array_to_img(fake[0])
            img.save(f"gen_imgs/generated_img{epoch}_{index}_.png")
            print("saved img")

        #Training discriminator: maximising log(D(x))+log(1 - D(G(z))

        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss(tf.ones((batch_size,1)), discriminator(real))
            loss_disc_fake = loss(tf.zeros((batch_size,1)),discriminator(fake))
            loss_disc = (loss_disc_real+loss_disc_fake)/2

        grads = disc_tape.gradient(loss_disc,discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads,discriminator.trainable_weights)
        )

        #Train generator
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss(tf.ones(batch_size,1),output)

        grads = gen_tape.gradient(loss_gen,generator.trainable_weights)

        opt_gen.apply_gradients(
            zip(grads,generator.trainable_weights)
        )


