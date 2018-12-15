from labm8 import fs
import task_utils
import rgx_utils as rgx
import pickle
from sklearn.utils import resample
import os
import numpy as np
import tensorflow as tf
import math
import struct
from keras import utils
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from absl import app
from absl import flags
import re
from keras import backend as K

flags.DEFINE_string('input_data', 'data/ir', 'Path to input data')
flags.DEFINE_string('out', 'model/', 'Path to folder in which to write saved Keras models and predictions')
flags.DEFINE_integer('num_epochs', 50, 'number of training epochs')
flags.DEFINE_integer('batch_size', 64, 'training batch size')
flags.DEFINE_integer('dense_layer', 32, 'dense layer size')
flags.DEFINE_integer('train_samples', 500, 'Number of training samples per class')
flags.DEFINE_integer('vsamples', 0, 'Sampling on validation set')
flags.DEFINE_integer('save_every', 100, 'Save checkpoint every N batches')
flags.DEFINE_integer('ring_size', 5, 'Checkpoint ring buffer length')
flags.DEFINE_bool('print_summary', False, 'Print summary of Keras model')

FLAGS = flags.FLAGS


########################################################################################################################
# Utils
########################################################################################################################
def get_onehot(y, num_classes):

    hot = np.zeros((len(y), num_classes), dtype=np.int32)
    for i, c in enumerate(y):
        hot[i][int(c) - 1] = 1

    return hot


def encode_srcs(input_files, dataset_name, unk_index):
    
    num_files = len(input_files)
    num_unks = 0
    seq_lengths = list()

    print('\n--- Preparing to read', num_files, 'input files for', dataset_name, 'data set')
    seqs = list()
    for i, file in enumerate(input_files):
        if i % 10000 == 0:
            print('\tRead', i, 'files')
        file = file.replace('.ll', '_seq.rec')
        assert os.path.exists(file), 'input file not found: ' + file
        with open(file, 'rb') as f:
            full_seq = f.read()
        seq = list()
        for j in range(0, len(full_seq), 4):  # read 4 bytes at a time
            seq.append(struct.unpack('I', full_seq[j:j + 4])[0])
        assert len(seq) > 0, 'Found empty file: ' + file
        num_unks += seq.count(str(unk_index))
        seq_lengths.append(len(seq))
        seqs.append([int(s) for s in seq])
    maxlen = max(seq_lengths)


    return seqs, maxlen


def pad_src(seqs, maxlen, unk_index):
    from keras.preprocessing.sequence import pad_sequences

    encoded = np.array(pad_sequences(seqs, maxlen=maxlen, value=unk_index))
    return np.vstack([np.expand_dims(x, axis=0) for x in encoded])


class   EmbeddingSequence(utils.Sequence):
    def __init__(self, batch_size, x_seq, y_1hot, embedding_mat):
        self.batch_size = batch_size
        self.num_samples = np.shape(x_seq)[0]
        self.dataset_len = int(self.num_samples // self.batch_size)
        self.x_seq = x_seq
        self.y_1hot = y_1hot
        self.emb = embedding_mat
        self.sess = tf.Session()
        self._set_index_array()

    def _set_index_array(self):
        self.index_array = np.random.permutation(self.num_samples)
        x_seq2 = self.x_seq[self.index_array]
        y_1hot2 = self.y_1hot[self.index_array]
        self.x_seq = x_seq2
        self.y_1hot = y_1hot2

    def on_epoch_end(self):
        self._set_index_array()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size * idx, self.batch_size * (idx + 1)

        x = self.x_seq[idx_begin:idx_end]
        emb_x = tf.nn.embedding_lookup(self.emb, x).eval(session=self.sess)
        return emb_x, self.y_1hot[idx_begin:idx_end]


class EmbeddingPredictionSequence(utils.Sequence):
    def __init__(self, batch_size, x_seq, embedding_mat):
        self.batch_size = batch_size
        self.x_seq = x_seq
        self.dataset_len = int(np.shape(x_seq)[0] // self.batch_size)
        self.emb = embedding_mat
        self.sess = tf.Session()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size * idx, self.batch_size * (idx + 1)
        x = self.x_seq[idx_begin:idx_end]
        emb_x = tf.nn.embedding_lookup(self.emb, x).eval(session=self.sess)
        return emb_x


class WeightsSaver(Callback):
    def __init__(self, model, save_every, ring_size):
        self.model = model
        self.save_every = save_every
        self.ring_size = ring_size
        self.batch = 0
        self.ring = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.save_every == 0:
            name = FLAGS.out + '/weights%d.h5' % self.ring
            self.model.save_weights(name)
            self.ring = (self.ring + 1) % self.ring_size
        self.batch += 1


########################################################################################################################
# Model
########################################################################################################################
class Main_Model(object):
    __name__ = "Vulnerability Detection"
    __basename__ = "Vulnerability_Detection"

    def init(self, seed: int, maxlen: int, embedding_dim: int, num_classes: int, dense_layer_size: int):
        from keras.layers import Input, LSTM, Dense
        from keras.layers.normalization import BatchNormalization
        from keras.models import Model

        np.random.seed(seed)

        # Keras model
        inp = Input(shape=(maxlen, embedding_dim,), dtype="float32", name="code_in")
        x = LSTM(embedding_dim, implementation=1, return_sequences=True, name="lstm_1")(inp)
        x = LSTM(embedding_dim, implementation=1, name="lstm_2")(x)

        x = BatchNormalization()(x)
        x = Dense(dense_layer_size, activation="relu")(x)
        outputs = Dense(num_classes, activation="sigmoid")(x)


        def MCC(y_true, y_pred):
            y_pred_pos = K.round(K.clip(y_pred, 0, 1))
            y_pred_neg = 1 - y_pred_pos

            y_pos = K.round(K.clip(y_true, 0, 1))
            y_neg = 1 - y_pos

            tp = K.sum(y_pos * y_pred_pos)
            tn = K.sum(y_neg * y_pred_neg)

            fp = K.sum(y_neg * y_pred_pos)
            fn = K.sum(y_pos * y_pred_neg)

            numerator = (tp * tn - fp * fn)
            denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            return numerator / (denominator + K.epsilon())
        def f1(y_true, y_pred):
            def recall(y_true, y_pred):
                """Recall metric.

                Only computes a batch-wise average of recall.

                Computes the recall, a metric for multi-label classification of
                how many relevant items are selected.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                recall = true_positives / (possible_positives + K.epsilon())
                return recall

            def precision(y_true, y_pred):
                """Precision metric.

                Only computes a batch-wise average of precision.

                Computes the precision, a metric for multi-label classification of
                how many selected items are relevant.
                """
                true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                precision = true_positives / (predicted_positives + K.epsilon())
                return precision
    
            precision = precision(y_true, y_pred)
            recall = recall(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))

        self.model = Model(inputs=inp, outputs=outputs)
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=['accuracy',f1,MCC])
        print('\tbuilt Keras model')

    def save(self, outpath: str):
        self.model.save(outpath)

    def restore(self, inpath: str):
        from keras.models import load_model
        self.model = load_model(inpath)

    def train(self, sequences: np.array, y_1hot: np.array, sequences_val: np.array, y_1hot_val: np.array,
              verbose: bool, epochs: int, batch_size: int) -> None:
        self.model.fit(x=sequences, y=y_1hot, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=True,
                       validation_data=(sequences_val, y_1hot_val))

    def train_gen(self, train_generator: EmbeddingSequence, validation_generator: EmbeddingSequence,
                  verbose: bool, epochs: int) -> None:
        checkpoint = WeightsSaver(self.model, FLAGS.save_every, FLAGS.ring_size)

        try:
            tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

            self.model.fit_generator(train_generator, epochs=epochs, verbose=verbose,
                                     validation_data=validation_generator,
                                     shuffle=True, callbacks=[checkpoint,tensor_board])
        except KeyboardInterrupt:
            print('Ctrl-C detected, saving weights to file')
            self.model.save_weights('weights-kill.h5')

    def predict(self, sequences: np.array, batch_size: int) -> np.array:
        p = np.array(self.model.predict(sequences, batch_size=batch_size, verbose=0))  
        indices = [np.argmax(x) for x in p]
        return [i + 1 for i in indices]  

    def predict_gen(self, generator: EmbeddingSequence) -> np.array:
        p = np.array(self.model.predict_generator(generator, verbose=0)) 
        indices = [np.argmax(x) for x in p]
        return [i + 1 for i in indices]  


########################################################################################################################
# Evaluate
########################################################################################################################
def evaluate(model, embeddings, folder_data, samples_per_class, folder_results, dense_layer_size, print_summary,
             num_epochs, batch_size):
    # Set seed for reproducibility
    seed = 204

    ####################################################################################################################
    # Get data
    vsamples_per_class = FLAGS.vsamples
    folder_data = re.sub('ir', 'seq', folder_data)
    # Data acquisition
    num_classes = 2
    y_train = np.empty(0)  # training
    X_train = list()
    folder_data_train = folder_data + '_train'
    y_val = np.empty(0)  # validation
    X_val = list()
    folder_data_val = folder_data + '_val'
    y_test = np.empty(0)  # testing
    X_test = list()
    folder_data_test = folder_data + '_test'
    print('Getting file names for', num_classes, 'classes from folders:')
    print(folder_data_train)
    print(folder_data_val)
    print(folder_data_test)
    for i in [0,1]:  # loop over classes

        # training: Read data file names

        folder = os.path.join(folder_data_train, str(i))
        assert os.path.exists(folder), "Folder: " + folder + ' does not exist'
        print('\ttraining  : Read file names from folder ', folder)
        listing = os.listdir(folder + '/')
        seq_files = [os.path.join(folder, f) for f in listing if f[-4:] == '.rec']

        # training: Randomly pick programs
        assert len(seq_files) >= samples_per_class, "Cannot sample " + str(samples_per_class) + " from " + str(
            len(seq_files)) + " files found in " + folder
        X_train += resample(seq_files, replace=False, n_samples=samples_per_class, random_state=seed)
        y_train = np.concatenate([y_train, np.array([int(i)] * samples_per_class, dtype=np.int32)])

        # validation: Read data file names
        folder = os.path.join(folder_data_val, str(i))
        assert os.path.exists(folder), "Folder: " + folder + ' does not exist'
        print('\tvalidation: Read file names from folder ', folder)
        listing = os.listdir(folder + '/')
        seq_files = [os.path.join(folder, f) for f in listing if f[-4:] == '.rec']

        # validation: Randomly pick programs
        if vsamples_per_class > 0:
            assert len(seq_files) >= vsamples_per_class, "Cannot sample " + str(vsamples_per_class) + " from " + str(
                len(seq_files)) + " files found in " + folder
            X_val += resample(seq_files, replace=False, n_samples=vsamples_per_class, random_state=seed)
            y_val = np.concatenate([y_val, np.array([int(i)] * vsamples_per_class, dtype=np.int32)])
        else:
            assert len(seq_files) > 0, "No .rec files found in" + folder
            X_val += seq_files
            y_val = np.concatenate([y_val, np.array([int(i)] * len(seq_files), dtype=np.int32)])

        # test: Read data file names
        folder = os.path.join(folder_data_test, str(i))
        assert os.path.exists(folder), "Folder: " + folder + ' does not exist'
        print('\ttest      : Read file names from folder ', folder)
        listing = os.listdir(folder + '/')
        seq_files = [os.path.join(folder, f) for f in listing if f[-4:] == '.rec']
        assert len(seq_files) > 0, "No .rec files found in" + folder
        X_test += seq_files
        y_test = np.concatenate([y_test, np.array([int(i)] * len(seq_files), dtype=np.int32)])

    # Load dictionary and cutoff statements
    folder_vocabulary = FLAGS.vocabulary_dir
    dictionary_pickle = os.path.join(folder_vocabulary, 'dic_pickle')
    print('\tLoading dictionary from file', dictionary_pickle)
    with open(dictionary_pickle, 'rb') as f:
        dictionary = pickle.load(f)
    unk_index = dictionary[rgx.unknown_token]
    del dictionary

    # Encode source codes and get max. sequence length
    X_seq_train, maxlen_train = encode_srcs(X_train, 'training', unk_index)
    X_seq_val, maxlen_val = encode_srcs(X_val, 'validation', unk_index)
    X_seq_test, maxlen_test = encode_srcs(X_test, 'testing', unk_index)
    maxlen = max(maxlen_train, maxlen_test, maxlen_val)
    print('Max. sequence length overall:', maxlen)
    print('Padding sequences')
    X_seq_train = pad_src(X_seq_train, maxlen, unk_index)
    X_seq_val = pad_src(X_seq_val, maxlen, unk_index)
    X_seq_test = pad_src(X_seq_test, maxlen, unk_index)

    # Get one-hot vectors for classification
    y_1hot_train = get_onehot(y_train, num_classes)
    y_1hot_val = get_onehot(y_val, num_classes)

    ####################################################################################################################
    # Setup paths

    # Set up names paths
    model_name = model.__name__
    model_path = os.path.join(folder_results,
                              "detect_vulnerability/models/{}.model".format(model_name))
    predictions_path = os.path.join(folder_results,
                                    "detect_vulnerability/predictions/{}.result".format(model_name))

    # If predictions have already been made with these embeddings, load them
    if fs.exists(predictions_path):
        print("\tFound predictions in", predictions_path, ", skipping...")
        with open(predictions_path, 'rb') as infile:
            p = pickle.load(infile)

    else:  # could not find predictions already computed with these embeddings

        # Embeddings
        import tensorflow as tf  # for embeddings lookup
        embedding_matrix_normalized = tf.nn.l2_normalize(embeddings, axis=1)
        vocabulary_size, embedding_dimension = embedding_matrix_normalized.shape
        print('XSEQ:\n', X_seq_train)
        print('EMB:\n', embedding_matrix_normalized)

        gen_test = EmbeddingPredictionSequence(batch_size, X_seq_test, embedding_matrix_normalized)

        # If models have already been made with these embeddings, load them
        if fs.exists(model_path):
            print("\n\tFound trained model in", model_path, ", skipping...")
            model.restore(model_path)

        else:  # could not find models already computed with these embeddings

            gen_train = EmbeddingSequence(batch_size, X_seq_train, y_1hot_train, embedding_matrix_normalized)
            gen_val = EmbeddingSequence(batch_size, X_seq_val, y_1hot_val, embedding_matrix_normalized)

            ############################################################################################################
            # Train

            # Create a new model and train it
            print('\n--- Initializing model...')
            model.init(seed=seed,
                       maxlen=maxlen,
                       embedding_dim=int(embedding_dimension),
                       num_classes=num_classes,
                       dense_layer_size=dense_layer_size)
            if print_summary:
                model.model.summary()
            print('\n--- Training model...')
            model.train_gen(train_generator=gen_train,
                            validation_generator=gen_val,
                            verbose=True,
                            epochs=num_epochs)

            # Save the model
            fs.mkdir(fs.dirname(model_path))
            model.save(model_path)
            print('\tsaved model to', model_path)

        ################################################################################################################
        # Test

        # Test model
        print('\n--- Testing model...')
        p = model.predict_gen(generator=gen_test)[0]

        # cache the prediction
        fs.mkdir(fs.dirname(predictions_path))
        with open(predictions_path, 'wb') as outfile:
            pickle.dump(p, outfile)
        print('\tsaved predictions to', predictions_path)

    ####################################################################################################################
    # Return accuracy
    accuracy = p == y_test  # prediction accuracy
    return accuracy


########################################################################################################################
# Main
########################################################################################################################
def main(argv):
    del argv    # unused

    ####################################################################################################################
    # Setup
    # Get flag values
    embeddings = task_utils.get_embeddings()
    folder_results = FLAGS.out
    assert len(folder_results) > 0, "Please specify a path to the results folder using --folder_results"
    folder_data = FLAGS.input_data
    dense_layer_size = FLAGS.dense_layer
    print_summary = FLAGS.print_summary
    num_epochs = FLAGS.num_epochs
    batch_size = FLAGS.batch_size
    train_samples = FLAGS.train_samples
    # Acquire data
    if not os.path.exists(folder_data + '_train'):
        print("Error")

    r = task_utils.llvm_ir_to_trainable(folder_data + '_train')
    assert os.path.exists(folder_data + '_val'), "Folder not found: " + folder_data + '_val'
    task_utils.llvm_ir_to_trainable(folder_data + '_val')
    assert os.path.exists(folder_data + '_test'), "Folder not found: " + folder_data + '_test'
    task_utils.llvm_ir_to_trainable(folder_data + '_test')

    print(folder_data)
    # Create directories if they do not exist
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)
    if not os.path.exists(os.path.join(folder_results, "models")):
        os.makedirs(os.path.join(folder_results, "models"))
    if not os.path.exists(os.path.join(folder_results, "predictions")):
        os.makedirs(os.path.join(folder_results, "predictions"))


    ####################################################################################################################
    print("\nEvaluating Vulnerability using Inst2Vec ...")
    classifyapp_accuracy = evaluate(Main_Model(), embeddings, folder_data, train_samples, folder_results,
                                    dense_layer_size, print_summary, num_epochs, batch_size)

    ####################################################################################################################

if __name__ == '__main__':
    app.run(main)
