# -*- coding: utf-8 -*-

"""
Autoencoder with L2-1 regularization for neural networks

We use the TensorBoard to plot the loss.

How to run:
    python3.5 autoencoder_dblp.py --layer1 200 --layer2 0  --gpu 0 --dimension 80
    # Olny three layers, Input-Encoding-Output with input size 80 * 4 and encoding size 200

    python3.5 autoencoder_dblp.py --layer1 200 --layer2 10  --gpu 6 --dimension 8
    # Five layers, Input-Hidden-Encoding-Hidden-Output with inputsize 80 * 4, hidden size 200, encoding size 10
"""

import tensorflow as tf
import numpy as np
import argparse
from classification import initial_read,svm_cv
from svm_perf_train_pecent import read_embedding,link_prediction_multi,read_deepwalk
# Training Parameters
learningrate_ = 0.001
num_steps = 20000
batch_size = 3051
select_Range = 80
display_step = 500
examples_to_show = 10

# Network Parameters
# num_hidden_0 = 600
num_hidden_1 = 300 # 1st layer num features
num_hidden_2 = 150 # 2nd layer num features (the latent dim)



def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def data_prep(dataset_name, ground_truth_file, select_idxs, select_Range, multilabel_flag):
    # Concatenating all embedding from various meta-paths and prepare one-hot encoding for results
    # 1. Reading all the spectral_emed vector in
    embeddings = []
    selected_class = 'a'
    for i in select_idxs:
        u, s, _, _, truth_label, mapping = initial_read(ground_truth_file,False, i,selected_class,dataset_name,multilabel_flag)
        nonzero_idx = 0
        for i in range(len(s)):
            if s[s.shape[0]-i-1]>=10e-5:
                print (i, s[s.shape[0]-i-1])
                nonzero_idx = i
                break
        embeddings.append(u[:,s.shape[0]-nonzero_idx-select_Range-1:s.shape[0]-nonzero_idx-1])
        # embeddings.append(u[:,s.shape[0]-nonzero_idx-select_Range-1:])
        # embeddings.append(u)

    assert('truth_label' in locals())
    single_mp_size = embeddings[0].shape[1]
    select_idxs_set = set(select_idxs)
    # Read all embedding
    embedd_names = []
    with open('../metapath_%s.txt' % dataset_name ,'r') as meta_path:
        count = 0
        for line in meta_path:
            toks = line.replace('\n','')
            if count in select_idxs_set:
                embedd_names.append(count)
            count+=1
    # Concatenating
    X = np.asarray([])
    for single_embedding in embeddings:
        if X.shape[0] == 0:
            X = single_embedding
        else:
            X = np.concatenate((X, single_embedding), axis = 1)
    Y = X
    return X,Y,truth_label,embedd_names, mapping

def output_embedding(encoding, mapping, filename):
    directory = 'compressed_embedding'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(directory + '/' + filename,'w') as output_file:
        output_file.write(str(encoding.shape[0]) + ' '+str(encoding.shape[1]) + '\n')
        for i in range(encoding.shape[0]):
            if i not in mapping: continue
            slice_row = str(mapping[i]) + ' ' + ' '.join([str(node) for node in encoding[i,:].tolist()]) + ' \n'
            output_file.write(slice_row)

def main():

    # Reset everything
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        help="Select a dataset: dblp (only supported so far)",
                        default = 'dblp')
    parser.add_argument("--class_type", type=bool,
                        help="Multi label classification or NOT",
                        default = False)
    parser.add_argument("--layer1", type=int, default=300,
                        help="The size of first Hidden layer")
    parser.add_argument("--layer2", type=int, default=0,
                        help="The size of second Hidden layer")
    parser.add_argument("--layer3", type=int, default=0,
                        help="The size of third Hidden layer")
    parser.add_argument("--leakyrate", type=float, default=0.9,
                        help="The leaky rate for leaky Relu activation.")
    parser.add_argument("--gpu", type=str, default="0",
                        help="The index of the GPU")
    parser.add_argument("--dimension", type=int, default=150,
                        help="The dimension for single embedding before concatenation")
    parser.add_argument("--externalemb", type=str, default='../ESim/results/imdb_combine_two_neg150_samplereal05__200.dat',
                        help="The path for external embedding")
    parser.add_argument("--externalhine", type=str, default='/shared/data/feng36/baseline/HINE/embed_imdb_200.txt',
                        help="The path for external embedding")
    parser.add_argument("--externalaspem", type=str, default='/shared/data/feng36/baseline/aspem/embed_imdb_200.txt',
                        help="The path for external embedding")
    parser.add_argument("--externalhin2vec", type=str, default='/shared/data/feng36/baseline/hin2vec/embed_imdb_200.txt',
                        help="The path for external embedding")
    parser.add_argument("--externalmetapath2vec", type=str, default='/shared/data/feng36/baseline/metapath2vec/embed_imdb_200.txt',
                        help="The path for external embedding")
    parser.add_argument("--externalPTE", type=str, default='/shared/data/feng36/baseline/PTE/PTE-imdb/context-movie-0.txt',
                        help="The path for external embedding")
    parser.add_argument("--linear", type=bool,
                        help="Multi label classification or NOT",
                        default = False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    tf.reset_default_graph()
    num_hidden_1 = args.layer1 # 1st layer num features
    num_hidden_2 = args.layer2 # 2nd layer num features (the latent dim)
    num_hidden_3 = args.layer3 # 3rd layer num features (the latent dim)
    dataset_name = args.dataset
    select_Range = args.dimension
    # The directory to save TensorBoard summaries
    from datetime import datetime
    now = datetime.now()
    logdir = "summaries/" + now.strftime("%Y%m%d-%H%M%S") + "/"
    print(num_steps)
    # Preprocess the inputs to be in [-1,1] and split the data in train/test sets
    # from sklearn import preprocessing, model_selection
    X,Y,truth_label,embedd_names,mapping = data_prep(dataset_name,'../practice/data/%s/groundtruth/name-label.txt' % dataset_name,[2,3,6,7],\
                                            # [(-500,-350),(-500,-350),(-500,-350),(-500,-350),(-500,-350),(-500,-350),(-1200,-1050),(-1100,-950),(-1100,-950)])
                                            select_Range,args.class_type)
    score = svm_cv(X, truth_label, args.class_type)
    print('Input: Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Macro-F1:',np.mean(score['test_f1_weighted']),'std:',np.std(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))

                                            # [(-500,-350)])
    # print('Before ae link_prediction:',link_prediction_multi(X,truth_label))
    # embedding_dim = 200
    # esim_embedding = read_embedding(args.externalemb,False,200,False)
    # hin2vec_embedding = read_embedding(args.externalhin2vec,True,embedding_dim,True)
    # aspem_embedding = read_embedding(args.externalaspem,False,embedding_dim,True)
    # hine_embedding = read_embedding(args.externalhine,False,embedding_dim,True)
    # metapath2vec_embedding = read_embedding(args.externalmetapath2vec,False,embedding_dim,False)
    # # metapath2vec_embedding = hine_embedding
    # pte_embedding = read_embedding(args.externalPTE,False,embedding_dim,True)
    # reverse_mapping = {}
    # for key in mapping:
    #     reverse_mapping[mapping[key]] = key
    # deepwalk_emb = read_deepwalk(100, ['mam','mum'],dataset_name,mapping=reverse_mapping)
    # print(len(deepwalk_emb))
    # esim_tmp,hin2vec_tmp,hine_tmp,aspem_tmp,deepwalk_tmp,m2v_tmp,pte_tmp = [],[],[],[],[],[],[]
    # m2v_misscount = 0
    # for num in range(1360):
    #     if num not in mapping: continue
    #     esim_tmp.append(esim_embedding[mapping[num]][0])
    #     hin2vec_tmp.append(hin2vec_embedding[mapping[num]][0])
    #     aspem_tmp.append(aspem_embedding[mapping[num]][0])
    #     pte_tmp.append(pte_embedding[mapping[num]][0])
    #     if mapping[num] not in metapath2vec_embedding:
    #         m2v_tmp.append(np.random.rand(int(embedding_dim)).tolist())
    #         m2v_misscount+=1
    #     else:
    #         m2v_tmp.append(metapath2vec_embedding[mapping[num]][0])
    #     if mapping[num] not in hine_embedding:
    #         hine_tmp.append(np.zeros(int(embedding_dim)).tolist())
    #     else:
    #         hine_tmp.append(hine_embedding[mapping[num]][0])
    #     if num not in deepwalk_emb: continue
    #     deepwalk_tmp.append(deepwalk_emb[num][0])
    # print('Misssing count of metapath2vec:',m2v_misscount)
    # esim_embedding = np.asarray(esim_tmp)
    # deepwalk_emb = np.asarray(deepwalk_tmp)
    # hin2vec_embedding = np.asarray(hin2vec_tmp)
    # hine_embedding = np.asarray(hine_tmp)
    # aspem_embedding = np.asarray(aspem_tmp)
    # metapath2vec_embedding = np.asarray(m2v_tmp)
    # pte_embedding = np.asarray(pte_tmp)
    # print(deepwalk_emb.shape)
    # # print('ESim link_prediction:',link_prediction_multi(esim_embedding,truth_label))
    # print('PTE link_prediction:',link_prediction_multi(pte_embedding,truth_label))
    # print('Deepwalk link_prediction:',link_prediction_multi(deepwalk_emb,truth_label))
    # print('Hin2vec link_prediction:',link_prediction_multi(hin2vec_embedding,truth_label))
    # print(hine_embedding.shape)
    # print('HINE link_prediction:',link_prediction_multi(hine_embedding,truth_label))
    # print('Metapath2vec link_prediction:',link_prediction_multi(metapath2vec_embedding,truth_label))
    # # print('Aspem link_prediction:',link_prediction_multi(aspem_embedding,truth_label))
    # labelcount = []
    # for lj in range(truth_label.shape[1]):
    #     labelcount.append(np.sum(truth_label[:,lj]))
    # print(labelcount)
    # score = svm_cv(pte_embedding, truth_label, args.class_type)
    # print('PTE:Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('MlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    #
    # score = svm_cv(metapath2vec_embedding, truth_label, args.class_type)
    # print('Metapath2vec:Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('MlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    # score = svm_cv(hin2vec_embedding, truth_label, args.class_type)
    # print('Hin2vec:Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('MlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    # score = svm_cv(hine_embedding, truth_label, args.class_type)
    # print('HINE:Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('MlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    # score = svm_cv(aspem_embedding, truth_label, args.class_type)
    # print('Aspem:Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('MlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    # score = svm_cv(X, truth_label, args.class_type)
    # print('Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('MlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    # score = svm_cv(deepwalk_emb, truth_label, args.class_type)
    # print('DeepwalkMacro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('DeepwalkMlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    # score = svm_cv(esim_embedding, truth_label, args.class_type)
    # print('EsimMacro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
    # print('EsimMlKNN_f1_macro',np.mean(score['MlKNN_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard-F1:',np.mean(score['MlKNN_Jaccard']))
    X_trn,y_trn = X,Y
    print(y_trn.shape)
    # Placeholders for input and output
    x = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name='input_emb')
    # encoding =
    d = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name='recover_emb')
    keep_prob = tf.placeholder(tf.float32)
    print('X shape:',X.shape)

    if num_hidden_3!=0:
        weights = {
            'encoder_h0': tf.Variable(tf.random_normal([X.shape[1], num_hidden_1],stddev=0.01)),
            'encoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],stddev=0.01)),
            'encoder_h2': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3],stddev=0.01)),
            'decoder_h0': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2],stddev=0.01)),
            'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1],stddev=0.01)),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, X.shape[1]],stddev=0.01)),
        }
        biases = {
            'encoder_b0': tf.Variable(tf.random_normal([num_hidden_1],stddev=0.01)),
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_2],stddev=0.01)),
            'encoder_b2': tf.Variable(tf.random_normal([num_hidden_3],stddev=0.01)),
            'decoder_b0': tf.Variable(tf.random_normal([num_hidden_2],stddev=0.01)),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1],stddev=0.01)),
            'decoder_b2': tf.Variable(tf.random_normal([X.shape[1]],stddev=0.01)),
        }

        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            print(weights['encoder_h1'])

            layer_0 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h0']),
                                           biases['encoder_b0']))

            layer_1 = tf.nn.tanh(tf.add(tf.matmul(layer_0, weights['encoder_h1']),
                                           biases['encoder_b1']))
            # Encoder Hidden layer with sigmoid activation #2
            print('after layper1')
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                           biases['encoder_b2']))
            return layer_2


        # Building the decoder
        def decoder(x):
            # Decoder Hidden layer with sigmoid activation #1
            layer_0 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h0']),
                                           biases['decoder_b0']))
            layer_1 = tf.nn.tanh(tf.add(tf.matmul(layer_0, weights['decoder_h1']),
                                           biases['decoder_b1']))
            # Decoder Hidden layer with sigmoid activation #2
            layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                           biases['decoder_b2']))
            return layer_2

    elif num_hidden_2 != 0:
        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([X.shape[1], num_hidden_1],stddev=0.01)),
            'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2],stddev=0.01)),
            'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1],stddev=0.01)),
            'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, X.shape[1]],stddev=0.01)),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1],stddev=0.01)),
            'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2],stddev=0.01)),
            'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1],stddev=0.01)),
            'decoder_b2': tf.Variable(tf.random_normal([X.shape[1]],stddev=0.01)),
        }

        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            print(weights['encoder_h1'])
            if not args.linear:
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                                        biases['encoder_b1']))
                # Encoder Hidden layer with sigmoid activation #2
                print('after layper1')
                layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                                        biases['encoder_b2']))
            else:
                layer_1 = tf.add(tf.matmul(x, weights['encoder_h1']),
                                                        biases['encoder_b1'])
                # Encoder Hidden layer with sigmoid activation #2
                print('after layper1')
                layer_2 = tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                                        biases['encoder_b2'])
            return layer_2


        # Building the decoder
        def decoder(x):
            # Decoder Hidden layer with sigmoid activation #1
            if not args.linear:
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                                        biases['decoder_b1']))
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                                        biases['decoder_b2']))
            else:
                layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']),
                                                        biases['decoder_b1'])
                # Decoder Hidden layer with sigmoid activation #2
                layer_2 = tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                                        biases['decoder_b2'])
            return layer_2
    else:

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([X.shape[1], num_hidden_1],stddev=0.01)),
            'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, X.shape[1]],stddev=0.01)),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1],stddev=0.01)),
            'decoder_b1': tf.Variable(tf.random_normal([X.shape[1]],stddev=0.01)),
        }

        # Building the encoder
        def encoder(x):
            # Encoder Hidden layer with sigmoid activation #1
            print(weights['encoder_h1'])
            '''Non-linear leaky'''
            if not args.linear:
                # layer_1 = lrelu(tf.add(tf.matmul(x, weights['encoder_h1']),
                #                                 biases['encoder_b1']),0.9)
                '''Non-linear tanh'''
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']),
                                               biases['encoder_b1']))
            else:
                '''linear'''
                layer_1 = tf.add(tf.matmul(x, weights['encoder_h1']),
                                                    biases['encoder_b1'])

            return tf.nn.dropout(layer_1, keep_prob)


        # Building the decoder
        def decoder(x):
            # Decoder Hidden layer with sigmoid activation #1
            '''Non-linear leaky'''
            if not args.linear:
                # layer_1 = lrelu(tf.add(tf.matmul(x, weights['decoder_h1']),
                #                                 biases['decoder_b1']),0.9)
                '''Non-linear tanh'''
                layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']),
                                               biases['decoder_b1']))
            else:
                '''linear'''
                layer_1 = tf.add(tf.matmul(x, weights['decoder_h1']),
                                                    biases['decoder_b1'])

            return tf.nn.dropout(layer_1, keep_prob)
    # Helper function to generate a layer
    print('Before creating encoder')
    encoder_op = encoder(x)
    decoder_op = decoder(encoder_op)
    d = decoder_op
    print('after decoder op')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    learningrate = learningrate_
    # Define the error function
    loss = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(i))) for i in tf.split(tf.subtract(x,d), num_or_size_splits=len(embedd_names), axis=1)])
    loss_block = [tf.sqrt(tf.reduce_sum(tf.square(i))) for i in tf.split(tf.subtract(x,d), num_or_size_splits=len(embedd_names), axis=1)]
    loss_block_l2 = [tf.reduce_sum(tf.square(i)) for i in tf.split(tf.subtract(x,d), num_or_size_splits=len(embedd_names), axis=1)]
    loss_1 = tf.reduce_sum(tf.norm(tf.subtract(x,d)))
    loss_summary = tf.summary.scalar('loss', loss)
    new_features = [loss_summary]
    v = tf.trainable_variables()
    print(len(new_features))
    merged = tf.summary.merge(new_features)
    variables_names = [v.name for v in tf.trainable_variables()]
    ''' GPU version '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth = False)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)) as sess:
        ''' CPU version'''
    # with tf.Session() as sess:
        # Initialize the summary writer
        train_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        with tf.name_scope('train'):
            # Training function
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        feed_prob = 1.0
        for i in range(num_steps):
            if i % 5000 == 0:
                if args.linear: print('Linear')
                print('Process: %s percent' % str((i /1000)))
                encoding = sess.run(encoder_op, feed_dict={x: X,keep_prob:1.0})
                score = svm_cv(encoding, truth_label, args.class_type)
                print('Encoding Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
                decoding = sess.run(decoder_op, feed_dict={x: X,keep_prob:1.0})
                score = svm_cv(decoding, truth_label, args.class_type)
                print('Decoding Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Weighted-F1:',np.mean(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
            # Take one training step
                # print('Encoding layer link_prediction:',link_prediction_multi(encoding,truth_label))
                # print('Decoding layer link_prediction:',link_prediction_multi(decoding,truth_label))
                print(learningrate)
            # if i %10000 == 0:learningrate = learningrate/10
            X_trn,_ = next_batch(batch_size,X,Y)
            X_trn = X
            summary, _, l,l1,l_block,l2_block= sess.run([merged, optimizer,loss,loss_1,loss_block,loss_block_l2], feed_dict={x: X_trn,learning_rate:learningrate,keep_prob:feed_prob})
            l,l1= sess.run([loss,loss_1], feed_dict={x: X,keep_prob:feed_prob})
            train_writer.add_summary(summary, i)
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f %f' % (i,l,l1))
                print('Block loss (L-2,1):',l_block)
                print('Block loss (L-2):',l2_block)
                print(learningrate)
        # Test Step
        encoding = sess.run(encoder_op, feed_dict={x: X,keep_prob:1.0})
        output_embedding(encoding, mapping, 'compressed_spectral_embedding_%s_%d.dat'
                                            %(dataset_name,encoding.shape[1]))
        print(encoding.shape)
        score = svm_cv(encoding, truth_label, args.class_type)
        print('Encoding Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),np.mean(score['test_f1_weighted']),'std:',np.std(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))
        print('Encoding layer link_prediction:',link_prediction_multi(encoding,truth_label))
        score = svm_cv(X, truth_label, args.class_type)
        print('Macro-F1:',np.mean(score['test_f1_macro']),'std:',np.std(score['test_f1_macro']),'Macro-F1:',np.mean(score['test_f1_weighted']),'std:',np.std(score['test_f1_weighted']),'Jaccard:',np.mean(score['Jaccard']),'std:',np.std(score['Jaccard']))

    train_writer.flush()
    train_writer.close()
    print(now.strftime("%Y%m%d-%H%M%S"))

if __name__ == '__main__':
    import os

    main()
