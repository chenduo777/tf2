import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from functools import partial
from utils.dataset import parse_fn
from utils.models import Generate, Discriminate
from utils.losses import generate_loss, discriminate_loss, gradient_penalty

batch_size = 16
lr = 0.0001
z_dim = 128
n_dis = 5
dataset = 'celeb_a'
gradient_penalty_weight = 10.0

train_dataset, info = tfds.load(dataset, split='train+validation+test', with_info=True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_data = train_dataset.shuffle(1000)
train_data = train_data.map(parse_fn, num_parallel_calls=AUTOTUNE)
train_data = train_data.batch(batch_size, drop_remainder=True)
train_data = train_data.prefetch(AUTOTUNE)

generate = Generate(input_shape=(1, 1, z_dim))
discriminate = Discriminate((64,64,3))

g_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)
d_opt = tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9)

@tf.function
def train_generate():
    with tf.GradientTape() as tape:
        random_vector = tf.random.normal((batch_size, 1, 1, z_dim))
        fake_img = generate(random_vector, training=True)
        fake_logit = discriminate(fake_img, training=True)
        g_loss = generate_loss(fake_logit)

    gradients = tape.gradient(g_loss, generate.trainable_variables)
    g_opt.apply_gradients(zip(gradients, generate.trainable_variables))
    return g_loss


@tf.function
def train_discriminate(real_img):
    with tf.GradientTape() as tape:
        random_vector = tf.random.normal((batch_size, 1, 1, z_dim))
        fake_img = generate(random_vector, training=True)
        real_logit = discriminate(real_img, training=True)
        fake_logit = discriminate(fake_img, training=True)
        f_loss, r_loss = discriminate_loss(real_logit, fake_logit)
        gp_loss = gradient_penalty(partial(discriminate, training=True), real_img, fake_img)
        d_loss = f_loss - r_loss + gp_loss * gradient_penalty_weight

    d_gradients = tape.gradient(d_loss, discriminate.trainable_variables)
    d_opt.apply_gradients(zip(d_gradients, discriminate.trainable_variables))
    return r_loss + f_loss, gp_loss

def combie_img(imgs, col=10, row=10):
    imgs = (imgs + 1) / 2
    imgs = imgs.numpy()
    b, h, w, _ = imgs.shape
    imgs_combine = np.zeros((h*row, w*col, 3))
    for i in range(row):
        for j in range(col):
            imgs_combine[i*h:(i+1)*h, j*w:(j+1)*w, :] = imgs[i*col+j]
    return imgs_combine

def train_wgan():
    log_dir = 'logs_wgan'
    model_dir = log_dir + '/models/'
    os.makedirs(model_dir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(log_dir)

    sample_random_vector = tf.random.normal((100, 1, 1, z_dim))
    for epoch in range(25):
        for step, real_img in enumerate(train_data):
            d_loss, gp_loss = train_discriminate(real_img)
            with summary_writer.as_default():
                tf.summary.scalar('d_loss', d_loss, d_opt.iterations)
                tf.summary.scalar('gp_loss', gp_loss, d_opt.iterations)

            if d_opt.iterations % n_dis == 0:
                g_loss = train_generate()
                with summary_writer.as_default():
                    tf.summary.scalar('g_loss', g_loss, g_opt.iterations)
                print('G loss: {:.2f} D loss: {:.2f}, GP loss: {:.2f}'.format(g_loss, d_loss, gp_loss))

            if g_opt.iterations % 100 == 0:
                x_fake = generate(sample_random_vector, training=False)
                save_img = combie_img(x_fake)
                with summary_writer.as_default():
                    tf.summary.image('generate', [save_img], g_opt.iterations)

        if epoch != 0:
            generate.save_weights(model_dir + 'generate_{}.h5'.format(epoch))

if __name__ == '__main__':
    train_wgan()


