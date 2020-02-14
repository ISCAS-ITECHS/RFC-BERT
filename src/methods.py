import tensorflow as tf


def w_plus_b(field_input_ids_embedding, hidden_size, output_layer, activity=None):
    Weights = tf.Variable(tf.random_normal([field_input_ids_embedding.shape[-1].value, hidden_size]))
    biases = tf.Variable(tf.zeros([1, hidden_size]) + 0.1)  # initial with 0.1
    Wx_plus_b = tf.matmul(field_input_ids_embedding, Weights) + biases
    if activity:
        Wx_plus_b = activity(tf.nn.dropout(Wx_plus_b, keep_prob=0.9))
    else:
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=0.9)
    return Wx_plus_b + output_layer


def NN(output_layer, field_input_ids_embedding):
    '''
    
    :param output_layer: BERT输出，batch_size * hidden_size
    :param field_input_ids_embedding: field部分内容
    :return:
    '''
    hidden_size = output_layer.shape[-1].value
    # field_input_ids_embedding = tf.layers.dense(field_input_ids_embedding, hidden_size)
    field_input_ids_embedding = tf.reshape(field_input_ids_embedding, [-1, field_input_ids_embedding.shape.as_list()[1]
                                                                       * field_input_ids_embedding.shape.as_list()[2]])
    return w_plus_b(field_input_ids_embedding, hidden_size, output_layer, activity=tf.nn.relu)


def CNN(output_layer, field_input_ids_embedding):
    hidden_size = output_layer.shape[-1].value
    x_image = tf.reshape(field_input_ids_embedding, [-1, field_input_ids_embedding.shape.as_list()[1],
                                                     field_input_ids_embedding.shape.as_list()[2], 1])
    w_conv = tf.Variable(tf.truncated_normal([3, hidden_size, 1, 3], stddev=0.1))
    b_conv = tf.Variable(tf.constant(0.1, shape=[3]))
    h_conv = tf.nn.relu(tf.nn.conv2d(x_image, w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)
    cnn_output = tf.nn.dropout(tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),
                               keep_prob=0.9)
    field_input_ids_embedding = tf.reshape(cnn_output, [-1, cnn_output.shape.as_list()[1]
                                                        * cnn_output.shape.as_list()[2]
                                                        * cnn_output.shape.as_list()[3]])
    return w_plus_b(field_input_ids_embedding, hidden_size, output_layer)


def Bi_GRU(output_layer, field_input_ids_embedding):  
    hidden_size = output_layer.shape[-1].value
    gru_cell_triple = tf.nn.rnn_cell.GRUCell(hidden_size)
    gru_outputs, gru_output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell_triple, gru_cell_triple,
                                                                           field_input_ids_embedding,
                                                                           dtype=tf.float32,
                                                                           swap_memory=True)
    encoded_gru_outputs = tf.concat(gru_outputs, 2)
    field_input_ids_embedding = tf.reshape(encoded_gru_outputs, [-1, encoded_gru_outputs.shape.as_list()[1]
                                                                 * encoded_gru_outputs.shape.as_list()[2]])
    return w_plus_b(field_input_ids_embedding, hidden_size, output_layer)



