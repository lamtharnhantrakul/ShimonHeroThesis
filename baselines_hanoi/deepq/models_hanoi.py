import tensorflow as tf
import tensorflow.contrib.layers as layers

'''
Builds ontop of OpenAI Baseline implementation of DQN and Duelling DQN with
modifications for bimanual, trimanual and tetramanual control
'''

def _bimanual_cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        # We will perform graph operations on the variable `out`
        out = inpt

        # Apply the layers of convolution to the input according to specified `conv`. For example `convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)]``
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)

        # Flatten the convolution layers to pass into MLP
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out

            # `hidden` specifies the layers of MLP. For example `hiddens=[256]`
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)

                # `layer_norm=False` by default
                # This is based on the idea of Layer Normalization on https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)

                # Apply relu activation to each of the MLP layer
                action_out = tf.nn.relu(action_out)

            # No activation function, this is a linear output representing the Q value associated with each action
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            # a dueling DQN splits the fully connected layer into two streams for Q and V
            with tf.variable_scope("state_value"):
                # In the above code we had `action_out = conv_out`, proceeded by FC layers.
                # Now we define `state_out = conv_out` since we want to use conv_out activations to determine a separate state value, independant of the action value
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)

                    # `layer_norm=False` by default
                    # This is based on the idea of Layer Normalization on https://www.tensorflow.org/api_docs/python/tf/contrib/layers/layer_norm
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)

                    # Very similar to above line `action_out = tf.nn.relu(action_out)`
                    state_out = tf.nn.relu(state_out)

                # Instead of outputting Q values per action, a DDQN ouputs one single state value, hence `num_outputs = 1`
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)

            # The following 3 steps is explained in the paper [https://arxiv.org/pdf/1511.06581.pdf] in equation (9)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)  # need to subtract the mean action scores from the mini-batch action scores (like mean centering)
            q_out = state_score + action_scores_centered #  verbatim from equation (9). Authors motivate that `q_out = state_score + action_scores` is actually not a good idea
        else:
            q_out = action_scores
        return q_out


def bimanual_cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _bimanual_cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)

def _cnn_to_mlp(convs, hiddens, dueling, inpt, num_actions, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

        if dueling:
            with tf.variable_scope("state_value"):
                state_out = conv_out
                for hidden in hiddens:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        state_out = layers.layer_norm(state_out, center=True, scale=True)
                    state_out = tf.nn.relu(state_out)
                state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            q_out = state_score + action_scores_centered
        else:
            q_out = action_scores
        return q_out


def cnn_to_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)
