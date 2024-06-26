import tensorflow as tf
from tensorflow import keras
def generate_loss(fake_logit):
    g_loss = tf.reduce_mean(fake_logit)
    return g_loss

def discriminate_loss(real_logit, fake_logit):
    r_loss = tf.reduce_mean(real_logit)
    f_loss = tf.reduce_mean(fake_logit)
    return f_loss, r_loss

def gradient_penalty(discriminator, real, fake):
    def _interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter
    x_img = _interpolate(real, fake)
    with tf.GradientTape() as tape:
        tape.watch(x_img)
        pred_logit = discriminator(x_img)
    grad = tape.gradient(pred_logit, x_img)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp_loss = tf.reduce_mean((norm - 1.)**2)
    return gp_loss

