import tensorflow as tf
import pygame

hello = tf.constant('Hello, ShimonHero!')
sess = tf.Session()
print(sess.run(hello))
