import tensorflow as tf

slim = tf.contrib.slim


@slim.add_arg_scope
def bottleneck0(inputs,depth,kernel=None,stride=1,scope=None):
	with tf.variable_scope(scope) as sc:
		shortcut=slim.conv2d(inputs,depth,[1,1],stride=stride,scope='shortcut')
	resnet = slim.conv2d(inputs, depth, kernel, stride=stride,scope='conv1')
	resnet = slim.conv2d(resnet, depth, kernel, stride=1,activation_fn=None,scope='conv2')
	output = tf.nn.relu(shortcut + resnet)
	return output
def bottleneck1(inputs,depth,kernel=None,stride=1,scope=None):
	with tf.variable_scope(scope) as sc:
		shortcut=inputs
	resnet = slim.conv2d(inputs, depth, kernel, stride=stride,scope='conv1')
	resnet = slim.conv2d(resnet, depth, kernel, stride=1,activation_fn=None,scope='conv2')
	output = tf.nn.relu(shortcut + resnet)
	return output



def inference(images, embedding_size=128, phase_train=True, weight_decay=0.0, reuse=None):
    batch_norm_params = {'decay': 0.995,'epsilon': 0.001,'updates_collections': None,'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ]}
    with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_initializer=tf.truncated_normal_initializer(stddev=0.1),weights_regularizer=slim.l2_regularizer(weight_decay),normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], is_training=True):
            with tf.variable_scope('resnet18') as sc:
                inputs=images
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,padding='SAME'):
                    net = slim.conv2d(inputs,64,3,stride=1,rate=1,scope='conv1')
                    print(net.shape)
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                    print(net.shape)
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu6,padding='SAME'):
                    with tf.variable_scope('block1') as sc:
                        with tf.variable_scope('unit1'):
                            net=bottleneck0(net,64,[3,3],stride=2,scope='bottleneck1')
                            print(net.shape)
                        with tf.variable_scope('unit2'):
                            net=bottleneck1(net,64,[3,3],stride=1,scope='bottleneck2')
                            print(net.shape)
                    with tf.variable_scope('block2') as sc:
                        with tf.variable_scope('unit1'):
                            net=bottleneck0(net,128,[3,3],stride=2,scope='bottleneck3')
                            print(net.shape)
                        with tf.variable_scope('unit2'):
                            net=bottleneck1(net,128,[3,3],stride=1,scope='bottleneck4')
                            print(net.shape)
                    with tf.variable_scope('block3') as sc:
                        with tf.variable_scope('unit1'):
                            net=bottleneck0(net,256,[3,3],stride=2,scope='bottleneck5')
                            print(net.shape)
                        with tf.variable_scope('unit2'):
                            net=bottleneck1(net,256,[3,3],stride=1,scope='bottleneck6')
                            print(net.shape)
                    with tf.variable_scope('block4') as sc:
                        with tf.variable_scope('unit1'):
                            net=bottleneck0(net,512,[3,3],stride=2,scope='bottleneck7')
                            print(net.shape)
                        with tf.variable_scope('unit2'):
                            net=bottleneck1(net,512,[3,3],stride=1,scope='bottleneck8')
                            print(net.shape)
                net = tf.reduce_mean(net, [1, 2], name='global_pool', keep_dims=True)
                print(net.shape)
                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                print(net.shape)
                net = slim.fully_connected(net, embedding_size, activation_fn=None, scope='Bottleneck', reuse=False)
                print(net.shape)
    return net





