from sogou_mrc.data.vocabulary import Vocabulary
from sogou_mrc.dataset.squad import SquadReader, SquadEvaluator
from sogou_mrc.model.bidaf import BiDAF
import tensorflow as tf
import logging
from sogou_mrc.data.batch_generator import BatchGenerator

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

data_folder = 'E:/dataset/SQuAD1.0/'
dev_file = data_folder + "dev-v1.1.json"

reader = SquadReader()
eval_data = reader.read(dev_file)
evaluator = SquadEvaluator(dev_file)

vocab = Vocabulary()
vocab_save_path = 'H:/result/bidaf/vocab.json'
vocab.load(vocab_save_path) # load vocab from save path

test_batch_generator = BatchGenerator(vocab, eval_data, batch_size=64)

save_dir = './save/best_weights/'
model = BiDAF(vocab)
model.load(save_dir)
model.session.run(tf.local_variables_initializer())
model.inference(test_batch_generator) # inference on test data