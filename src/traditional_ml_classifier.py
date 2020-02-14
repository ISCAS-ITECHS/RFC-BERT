# -- coding: utf-8 --

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from collections import defaultdict
import decimal
import csv
from bert import tokenization


max_seq_length = 56  
field_seq_length = 10
category_num = 9

def clip(x, top=category_num):
    return np.clip(x, 0, top) 


def getPRF(pred, real, category_num, margin=0.5): 
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)
    TN = defaultdict(int)
    total = real.shape[0]
    a_ac = a_pr = a_re = 0
    res = [0] * 2 * category_num
    effective_num = 0
    for p, r in zip(pred, real):
        if abs(p-r) <= margin:
            TP[int(r)] += 1
            effective_num += 1
            for i in range(0, category_num, 1):
                if int(r) != i:
                    TN[i] += 1
        else:
            p_int = int(p)
            t = decimal.Decimal(p_int)
            decimal.getcontext().rounding = decimal.ROUND_UP
            t = int(round(t, 0))
            FP[t] += 1
            FN[int(r)] += 1
    for i in range(0, category_num, 1):
        a_ac += (TP[i] + TN[i])/(TP[i] + FP[i] + FN[i] + TN[i]) if TP[i] + FP[i] + FN[i] + TN[i] != 0 else 0
        a_pr += TP[i]/(TP[i] + FP[i]) if TP[i] + FP[i] != 0 else 0
        res[i] += TP[i]/(TP[i] + FP[i]) if TP[i] + FP[i] != 0 else 0
        a_re += TP[i]/(TP[i] + FN[i]) if TP[i] + FN[i] != 0 else 0
        res[i + category_num] += TP[i]/(TP[i] + FN[i]) if TP[i] + FN[i] != 0 else 0
    numerator = 0
    denominator1 = denominator2 = 0
    for k, v in TP.items():
        numerator += v
    for k, v in FP.items():
        denominator1 += v
    for k, v in FN.items():
        denominator2 += v
    pral = numerator / (numerator + denominator1)
    real = numerator / (numerator + denominator2)
    acc = effective_num / total
    return a_ac / category_num, a_pr / category_num, a_re / category_num, res, pral, real, acc


def _read_tsv(input_file, quotechar=None):  
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def _create_examples(lines):  
    """Creates examples for the training and dev sets."""
    examples = []
    for line in lines:
        text_a = tokenization.convert_to_unicode(line[2].strip())
        text_field = tokenization.convert_to_unicode(line[1].strip())
        label = tokenization.convert_to_unicode(line[3])
        examples.append((text_a, text_field, label))
    return examples


def convert_single_example(example_tuple, label_map, max_seq_length, tokenizer, field_seq_length):  
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens_a = tokenizer.tokenize(example_tuple[1].strip())


    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_a:
        tokens.append(token)
    tokens.append("[SEP]")


    input_ids = tokenizer.convert_tokens_to_ids(tokens)


    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
    
    assert len(input_ids) == max_seq_length
    
    final_ids = []
    
    final_ids.extend(input_ids)
    label_id = label_map[example_tuple[2]]

    return final_ids, label_id


def svm(lr=0.002):
    
    x_data = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32)  # max_seq_length + field_seq_length
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    word_embedding_dictionary = tf.get_variable("embd_word", shape=[28997, 768],
                                                initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_data_coding = tf.nn.embedding_lookup(word_embedding_dictionary, x_data)
    A = tf.Variable(tf.random_normal(shape=[max_seq_length * 768, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))

    x_data_coding = tf.reshape(x_data_coding, [-1, max_seq_length * 768])
    model_output = tf.subtract(tf.matmul(x_data_coding, A), b)

    # Declare vector L2 'norm' function squared
    l2_norm = tf.reduce_sum(tf.square(A))

    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    alpha = tf.constant([0.01])
    classification_term = tf.reduce_mean(
        tf.maximum(0., tf.subtract(1., tf.multiply(model_output, tf.to_float(y_target)))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

    my_opt = tf.train.GradientDescentOptimizer(lr)
    train_step = my_opt.minimize(loss)
    return x_data, y_target, train_step, model_output, classification_term


def thesis_method_1(lr=0.00002, word_dim=768):
    x_data = tf.placeholder(shape=[None, max_seq_length], dtype=tf.int32)  
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    word_embedding_dictionary = tf.get_variable("embd_word", shape=[28997, word_dim],
                                                initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_data_coding = tf.nn.embedding_lookup(word_embedding_dictionary, x_data)

    gru_cell_triple = tf.nn.rnn_cell.GRUCell(word_dim)
    x_data_outputs, x_data_output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell_triple, gru_cell_triple,
                                                                           x_data_coding,
                                                                           dtype=tf.float32,
                                                                           swap_memory=True)
    encoded_x_data_outputs = tf.concat(x_data_outputs, 2)
    encoded_x_data_outputs = tf.reshape(encoded_x_data_outputs, [-1,
                                                                 encoded_x_data_outputs.shape.as_list()[1],
                                                                 encoded_x_data_outputs.shape.as_list()[2],
                                                                 1])
    pooling_output = tf.nn.dropout(tf.nn.max_pool(encoded_x_data_outputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),
                               keep_prob=0.9)
    pooling_output = tf.reshape(pooling_output,
                                [-1, pooling_output.shape.as_list()[1] * pooling_output.shape.as_list()[2]])
    A = tf.Variable(tf.random_normal(shape=[pooling_output.shape.as_list()[1], category_num]))
    b = tf.Variable(tf.random_normal(shape=[1, category_num]))
    l2_norm = tf.reduce_sum(tf.square(A))
    model_output = tf.matmul(pooling_output, A) + b
    loss, predict = softmax_layer(model_output, y_target, num_labels=category_num)
    alpha = tf.constant([0.01])
    loss = tf.add(loss, tf.multiply(alpha, l2_norm))/61
    # Declare vector L2 'norm' function squared
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    return x_data, y_target, train_op, predict, loss


def softmax_layer(logits, labels, num_labels):
    # total = logits.shape.as_list()[0]
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=one_hot_labels)
    loss = tf.reduce_sum(loss)
    # loss /= total

    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def NN(lr=0.00002, word_dim=768):
    x_data = tf.placeholder(shape=[None, max_seq_length],  
                            dtype=tf.int32) 
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    word_embedding_dictionary = tf.get_variable("embd_word", shape=[28997, word_dim],
                                                initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_data_coding = tf.nn.embedding_lookup(word_embedding_dictionary, x_data)
    x_data_coding = tf.reshape(x_data_coding,
                                [-1, x_data_coding.shape.as_list()[1] * x_data_coding.shape.as_list()[2]])
    A = tf.Variable(tf.random_normal(shape=[x_data_coding.shape.as_list()[1], category_num]))
    l2_norm = tf.reduce_sum(tf.square(A))
    b = tf.Variable(tf.random_normal(shape=[1, category_num]))
    model_output = tf.matmul(x_data_coding, A) + b
    model_output = tf.nn.dropout(model_output, keep_prob=0.9)
    loss, predict = softmax_layer(model_output, y_target, num_labels=category_num)
    alpha = tf.constant([0.01])
    loss = tf.add(loss, tf.multiply(alpha, l2_norm)) / 61
    # Declare vector L2 'norm' function squared
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    return x_data, y_target, train_op, predict, loss


def CNN(lr=0.00002, word_dim=768):
    x_data = tf.placeholder(shape=[None, max_seq_length],  
                            dtype=tf.int32)  
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    word_embedding_dictionary = tf.get_variable("embd_word", shape=[28997, word_dim],
                                                initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_data_coding = tf.nn.embedding_lookup(word_embedding_dictionary, x_data)
    x_image = tf.reshape(x_data_coding, [-1, x_data_coding.shape.as_list()[1],
                                         x_data_coding.shape.as_list()[2], 1])
    w_conv = tf.Variable(tf.truncated_normal([3, word_dim, 1, 3], stddev=0.1))
    b_conv = tf.Variable(tf.constant(0.1, shape=[3]))
    h_conv = tf.nn.relu(tf.nn.conv2d(x_image, w_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)
    cnn_output = tf.nn.dropout(tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'),
                               keep_prob=0.9)
    field_input_ids_embedding = tf.reshape(cnn_output, [-1, cnn_output.shape.as_list()[1]
                                                        * cnn_output.shape.as_list()[2]
                                                        * cnn_output.shape.as_list()[3]])
    A = tf.Variable(tf.random_normal([field_input_ids_embedding.shape[-1].value, category_num]))  
    b = tf.Variable(tf.zeros([1, category_num]) + 0.1)  # initial with 0.1
    l2_norm = tf.reduce_sum(tf.square(A))
    model_output = tf.matmul(field_input_ids_embedding, A) + b
    model_output = tf.nn.dropout(model_output, keep_prob=0.9)
    loss, predict = softmax_layer(model_output, y_target, num_labels=category_num)
    alpha = tf.constant([0.01])
    loss = tf.add(loss, tf.multiply(alpha, l2_norm)) / 61
    # Declare vector L2 'norm' function squared
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    return x_data, y_target, train_op, predict, loss


def RNN(lr=0.00002, word_dim=768):
    x_data = tf.placeholder(shape=[None, max_seq_length],  
                            dtype=tf.int32)  
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.int32)
    word_embedding_dictionary = tf.get_variable("embd_word", shape=[28997, word_dim],
                                                initializer=tf.truncated_normal_initializer(stddev=0.02))
    x_data_coding = tf.nn.embedding_lookup(word_embedding_dictionary, x_data)
    gru_cell_triple = tf.nn.rnn_cell.GRUCell(word_dim)
    gru_outputs, gru_output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell_triple, gru_cell_triple,
                                                                     x_data_coding,
                                                                     dtype=tf.float32,
                                                                     swap_memory=True)
    encoded_gru_outputs = tf.concat(gru_outputs, 2)
    field_input_ids_embedding = tf.reshape(encoded_gru_outputs, [-1, encoded_gru_outputs.shape.as_list()[1]
                                                                 * encoded_gru_outputs.shape.as_list()[2]])
    A = tf.Variable(tf.random_normal([field_input_ids_embedding.shape[-1].value, category_num]))  
    b = tf.Variable(tf.zeros([1, category_num]) + 0.1)  # initial with 0.1
    l2_norm = tf.reduce_sum(tf.square(A))
    model_output = tf.matmul(field_input_ids_embedding, A) + b
    model_output = tf.nn.dropout(model_output, keep_prob=0.9)
    loss, predict = softmax_layer(model_output, y_target, num_labels=category_num)
    alpha = tf.constant([0.01])
    loss = tf.add(loss, tf.multiply(alpha, l2_norm)) / 61
    # Declare vector L2 'norm' function squared
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
    return x_data, y_target, train_op, predict, loss


if __name__ == "__main__":
    tokenizer = tokenization.FullTokenizer(
        vocab_file='cased_L-12_H-768_A-12/vocab.txt',
        do_lower_case=True
    )
    train_lines = _create_examples(_read_tsv('rfc/11202/train.tsv'))
    test_lines = _create_examples(_read_tsv('rfc/11202/test.tsv'))
    label_list = [
        "100",  
        "210",  
        "220",  
        "230",  
        "231",  
        "240",  
        "301",  
        "400",  
        "500",  
    ]
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for example in train_lines:
        x_id, y_id = convert_single_example(example, label_map,
                                            max_seq_length=max_seq_length,
                                            tokenizer=tokenizer,
                                            field_seq_length=field_seq_length)
        x_train.append(x_id)
        y_train.append(y_id)
    for example in test_lines:
        x_id, y_id = convert_single_example(example, label_map,
                                            max_seq_length=max_seq_length,
                                            tokenizer=tokenizer,
                                            field_seq_length=field_seq_length)
        x_test.append(x_id)
        y_test.append(y_id)

    x_vals = np.array(x_train)
    y_vals = np.array(y_train)
    x_test_vals = np.array(x_test)
    y_test_vals = np.array(y_test)

    # define the name of current machine learning name

    # ml_name = 'md1'
    # ml_name = 'rnn'
    ml_name = 'cnn'
    # ml_name = 'nn'
    # ml_name = 'svm'
    batch_size = 100
    # x_data, y_target, train_step, model_output, classification_term = svm(lr=0.02)
    # x_data, y_target, train_step, model_output, classification_term = thesis_method_1(lr=0.02)
    # x_data, y_target, train_step, model_output, classification_term = NN(lr=0.02)
    x_data, y_target, train_step, model_output, classification_term = CNN(lr=0.02)
    # x_data, y_target, train_step, model_output, classification_term = RNN(lr=0.02)


    saver = tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training loop
        for i in range(10000):  #  
            rand_index = np.random.choice(len(x_vals), size=batch_size)
            rand_x = x_vals[rand_index]
            rand_y = np.transpose([y_vals[rand_index]])
            _ = sess.run([train_step], feed_dict={x_data: rand_x, y_target: rand_y})
            if i > 100:
                print("current:",end='')
                print(i)
                mo, ct = sess.run([model_output, classification_term],
                                  feed_dict={x_data: x_test_vals, y_target: np.reshape(y_test_vals, (np.size(y_test_vals), 1))})
                mo = np.squeeze(mo)
                Zmax, Zmin = mo.max(), mo.min()
                mo = ((mo - Zmin) / (Zmax - Zmin)) * category_num
                mo = clip(mo)
                a, p, r, _r, kp, kr, kac = getPRF(mo, y_test_vals, category_num=category_num)
                print("acc: ", end='')
                print(kac)
                print("p: ", end='')

                print(p)
                print("r: ", end='')

                print(r)
                print("loss: ", end='')
                print(ct)
        saver.save(sess, "./t_model/" + ml_name + "_model/" + ml_name + "_model.ckpt")
