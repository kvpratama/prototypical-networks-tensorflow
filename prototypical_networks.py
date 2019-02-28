
import os
import glob
from PIL import Image
import numpy as np
import tensorflow as tf

####### Load Omniglot Dataset
####### Following the procedure of Vinyals et al. (Matching Networks)
####### by resizing the grayscale images to 28 × 28
####### and augmenting the character classes with rotations in multiples of 90 degrees.
def load_omniglot(split_file='train.txt'):
    ### Load Dataset
    split_path = os.path.join(root_dir, 'splits', split_file)

    with open(split_path, 'r') as split:
        classes = [line.rstrip() for line in split.readlines()]

    n_classes = len(classes)
    ### Initialize dataset with a shape as number of classes, examples, height, and width
    dataset = np.zeros([n_classes, num_examples, img_height, img_width], dtype=np.float32)

    for label, name in enumerate(classes):
        alphabet, character, rotation = name.split('/')
        rotation = float(rotation[3:])
        img_dir = os.path.join(root_dir, 'data', alphabet, character)
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

        for index, img_file in enumerate(img_files):
            values = 1. - np.array(Image.open(img_file).rotate(rotation).resize((img_width, img_height)), np.float32, copy=False)
            dataset[label, index] = values

    print(dataset.shape)
    return dataset, n_classes

def convolution_block(inputs, out_channels, name='conv'):
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, padding='SAME')
        conv = tf.contrib.layers.batch_norm(conv, updates_collections=None, decay=0.99, scale=True, center=True)
        conv = tf.nn.relu(conv)
        conv = tf.contrib.layers.max_pool2d(conv, 2)
        return conv

def encoder(support_set, h_dim, z_dim, reuse=False):
    with tf.variable_scope('encoder', reuse=reuse):
        net = convolution_block(support_set, h_dim, name='conv_1')
        net = convolution_block(net, h_dim, name='conv_2')
        net = convolution_block(net, h_dim, name='conv_3')
        net = convolution_block(net, z_dim, name='conv_4')
        net = tf.contrib.layers.flatten(net)
        return net

def euclidean_distance(a, b):
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

root_dir = 'data/omniglot/'
num_examples = 20
img_width = 28
img_height = 28
channels = 1
num_way = 60 # number of classes
num_shot = 5 # number of examples per class for support set
num_query = 5
num_examples = 20
h_dim = 64
z_dim = 64
num_epochs = 20
num_episodes = 100

n_test_episodes = 1000
n_test_way = 20
n_test_shot = 5
n_test_query = 15

x_train, x_classes = load_omniglot()
x_test, x_test_classes = load_omniglot(split_file='test.txt')

support_set = tf.placeholder(tf.float32, [None, None, img_height, img_width, channels])
query_set = tf.placeholder(tf.float32, [None, None, img_height, img_width, channels])
support_set_shape = tf.shape(support_set)
query_set_shape = tf.shape(query_set)
num_classes, num_support_points = support_set_shape[0], support_set_shape[1]
num_query_points = query_set_shape[1]
y = tf.placeholder(tf.int64, [None, None])
y_one_hot = tf.one_hot(y, depth=num_classes)
support_set_embeddings = encoder(tf.reshape(support_set, [num_classes * num_support_points, img_height, img_width, channels]), h_dim, z_dim)
embedding_dimension = tf.shape(support_set_embeddings)[-1]
class_prototype = tf.reduce_mean(tf.reshape(support_set_embeddings, [num_classes, num_support_points, embedding_dimension]), axis=1)
query_set_embeddings = encoder(tf.reshape(query_set, [num_classes * num_query_points, img_height, img_width, channels]), h_dim, z_dim, reuse=True)
distance = euclidean_distance(query_set_embeddings, class_prototype)
predicted_probability = tf.reshape(tf.nn.log_softmax(-distance), [num_classes, num_query_points, -1])
loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, predicted_probability), axis=-1), [-1]))
accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(predicted_probability, axis=-1), y)))

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.InteractiveSession()
init_op = tf.global_variables_initializer()
sess.run(init_op)

for epoch in range(num_epochs):
    for episode in range(num_episodes):
        episodic_classes = np.random.permutation(x_classes)[:num_way]
        support = np.zeros([num_way, num_shot, img_height, img_width], dtype=np.float32)
        query = np.zeros([num_way, num_query, img_height, img_width], dtype=np.float32)

        for index, class_ in enumerate(episodic_classes):
            selected = np.random.permutation(num_examples)[:num_shot + num_query]
            support[index] = x_train[class_, selected[:num_shot]]
            query[index] = x_train[class_, selected[num_shot:]]

        support = np.expand_dims(support, axis=-1)
        query = np.expand_dims(query, axis=-1)
        labels = np.tile(np.arange(num_way)[:, np.newaxis], (1, num_query)).astype(np.uint8)
        _, loss_, accuracy_ = sess.run([train_op, loss, accuracy], feed_dict={support_set: support, query_set: query, y:labels})

        if (episode+1) % 10 == 0:
            print('Epoch {} : Episode {} : Loss: {}, Accuracy: {}'.format(epoch+1, episode+1, loss_, accuracy_))

print('Testing...')

avg_acc = 0.
for epi in range(n_test_episodes):
    epi_classes = np.random.permutation(x_test_classes)[:n_test_way]
    support = np.zeros([n_test_way, n_test_shot, img_height, img_width], dtype=np.float32)
    query = np.zeros([n_test_way, n_test_query, img_height, img_width], dtype=np.float32)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(num_examples)[:n_test_shot + n_test_query]
        support[i] = x_test[epi_cls, selected[:n_test_shot]]
        query[i] = x_test[epi_cls, selected[n_test_shot:]]
    support = np.expand_dims(support, axis=-1)
    query = np.expand_dims(query, axis=-1)
    labels = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
    ls, ac = sess.run([loss, accuracy], feed_dict={support_set: support, query_set: query, y:labels})
    avg_acc += ac
    if (epi+1) % 50 == 0:
        print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi+1, n_test_episodes, ls, ac))
avg_acc /= n_test_episodes
print('Average Test Accuracy: {:.5f}'.format(avg_acc))
