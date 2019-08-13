import time
import functools
from operator import mul
import numpy as np
import tensorflow as tf
from . import vgg
from . import transform
from . import utils


def optimize(content_targets,
             style_target,
             content_weight,
             style_weight,
             tv_weight,
             vgg_path,
             pooling='max',
             epochs=2,
             print_iterations=1000,
             batch_size=4,
             save_path='saver/fns.ckpt',
             slow=False,
             learning_rate=1e-3,
             debug=True):
    if slow:
        batch_size = 1
    mod = len(content_targets) % batch_size
    if mod > 0:
        content_targets = content_targets[:-mod]

    batch_shape = (batch_size, 256, 256, 3)
    style_shape = (1,) + style_target.shape
    style_features = {}
    vgg_weights, vgg_mean_pixel = vgg.load_net(vgg_path)

    # compute style features in feedforward mode
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
        style_image = tf.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image, vgg_mean_pixel)
        net = vgg.net_preloaded(vgg_weights, style_image_pre, pooling)
        style_pre = np.array([style_target])
        for layer in vgg.STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default(), tf.Session() as sess:
        # dataset
        num_examples = len(content_targets)
        dataset = tf.data.Dataset.from_tensor_slices(content_targets)
        # dataset = dataset.shuffle(num_examples)
        dataset = dataset.map(utils.parse_fn)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        X_batch = iterator.get_next()

        # content loss
        X_pre = vgg.preprocess(X_batch, vgg_mean_pixel)
        content_net = vgg.net_preloaded(vgg_weights, X_pre, pooling)
        content_features = {vgg.CONTENT_LAYER: content_net[vgg.CONTENT_LAYER]}

        if slow:
            preds = tf.Variable(tf.random_normal(X_batch.get_shape()) * 0.256)
            preds_pre = preds
        else:
            preds = transform.net(X_batch / 255.0, batch_size=batch_size)
            preds_pre = vgg.preprocess(preds, vgg_mean_pixel)
        net = vgg.net_preloaded(vgg_weights, preds_pre, pooling)

        content_size = _tensor_size(content_features[vgg.CONTENT_LAYER]) * batch_size
        content_loss = content_weight * (2 * tf.nn.l2_loss(net[vgg.CONTENT_LAYER] - content_features[vgg.CONTENT_LAYER]) / content_size)

        # style loss
        style_losses = []
        for style_layer in vgg.STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i: i.value, layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(feats, perm=[0, 2, 1])
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)
        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:, 1:, :, :])
        tv_x_size = _tensor_size(preds[:, :, 1:, :])
        y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
        x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
        tv_loss = tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / batch_size

        # total loss
        loss = content_loss + style_loss + tv_loss

        # overall loss
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            iterations = 0
            sess.run(iterator.initializer)
            while iterations * batch_size < num_examples:
                iterations += 1
                start_time = time.time()
                train_step.run()
                end_time = time.time()
                delta_time = end_time - start_time
                if debug:
                    print("epoch: %s, iterations: %s, batch time: %s" % (epoch, iterations, delta_time))

                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                is_last = epoch == epochs - 1 and iterations * batch_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    tup = sess.run([style_loss, content_loss, tv_loss, loss, preds])
                    _style_loss, _content_loss, _tv_loss, _loss, _preds = tup
                    losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                       _preds = vgg.unprocess(_preds, vgg_mean_pixel)
                    else:
                       saver = tf.train.Saver()
                       res = saver.save(sess, save_path)
                    yield(_preds, losses, iterations, epoch)


def _tensor_size(tensor):
    return functools.reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)
