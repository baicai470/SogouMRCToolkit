# coding: utf-8
from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.bidaf import BiDAF
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Prepare the dataset reader and evaluator
data_folder = 'E:/dataset/SQuAD1.0/'
embedding_folder = 'E:/dataset/glove/'

train_file = data_folder + "train-v1.1.json"
dev_file = data_folder + "dev-v1.1.json"

reader = SquadReader()
train_data = reader.read(train_file)
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

# Build a vocabulary and load the pretrained embedding
vocab = Vocabulary()
vocab.build_vocab(train_data + eval_data, min_word_count=3, min_char_count=10)
word_embedding = vocab.make_word_embedding(embedding_folder + "glove.6B.100d.txt")

# save vocab
vocab_save_path = 'H:/result/bidaf/vocab.json'
vocab.save(vocab_save_path)

# Use the feature extractor,which is only necessary when using linguistic features.
# feature_transformer = FeatureExtractor(features=['match_lemma','match_lower','pos','ner','context_tf'],
# build_vocab_feature_names=set(['pos','ner']),word_counter=vocab.get_word_counter())
# train_data = feature_transformer.fit_transform(dataset=train_data)
# eval_data = feature_transformer.transform(dataset=eval_data)

# Build a batch generator for training and evaluation,where additional features and a feature vocabulary are
# necessary when a linguistic feature is used.
train_batch_generator = BatchGenerator(vocab, train_data, batch_size=64, training=True)

eval_batch_generator = BatchGenerator(vocab, eval_data, batch_size=64)
# train and save checkpoint in save_dir
save_dir = 'H:/result/bidaf'
# Import the built-in model and compile the training operation, call functions such as train_and_evaluate for
# training and evaluation.
model = BiDAF(vocab, pretrained_word_embedding=word_embedding)
model.compile(tf.train.AdamOptimizer, 0.001)
model.train_and_evaluate(train_batch_generator, eval_batch_generator, evaluator, epochs=15, eposides=2,
                         save_dir=save_dir)


