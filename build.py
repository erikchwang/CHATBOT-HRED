import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *

LEARNING_RATE = tf.placeholder(dtype=tf.float32, shape=[])
DIALOGUE = tf.placeholder(dtype=tf.int32, shape=[None])
BOS_LOCATIONS = tf.placeholder(dtype=tf.int32, shape=[None])
EOS_LOCATIONS = tf.placeholder(dtype=tf.int32, shape=[None])
UPDATE = model_update(LEARNING_RATE, train_set_path, None)

VALIDATE_FULL_COUNT, VALIDATE_FULL_LOSS, VALIDATE_FULL_HIT, VALIDATE_TAIL_COUNT, VALIDATE_TAIL_LOSS, VALIDATE_TAIL_HIT = model_assess(
    validate_set_path, True
)

TEST_FULL_COUNT, TEST_FULL_LOSS, TEST_FULL_HIT, TEST_TAIL_COUNT, TEST_TAIL_LOSS, TEST_TAIL_HIT = model_assess(
    test_set_path, True
)

PREDICT = model_predict(DIALOGUE, BOS_LOCATIONS, EOS_LOCATIONS, True)

tf.add_to_collection(name="LEARNING_RATE", value=LEARNING_RATE)
tf.add_to_collection(name="DIALOGUE", value=DIALOGUE)
tf.add_to_collection(name="BOS_LOCATIONS", value=BOS_LOCATIONS)
tf.add_to_collection(name="EOS_LOCATIONS", value=EOS_LOCATIONS)
tf.add_to_collection(name="UPDATE", value=UPDATE)
tf.add_to_collection(name="VALIDATE_FULL_COUNT", value=VALIDATE_FULL_COUNT)
tf.add_to_collection(name="VALIDATE_FULL_LOSS", value=VALIDATE_FULL_LOSS)
tf.add_to_collection(name="VALIDATE_FULL_HIT", value=VALIDATE_FULL_HIT)
tf.add_to_collection(name="VALIDATE_TAIL_COUNT", value=VALIDATE_TAIL_COUNT)
tf.add_to_collection(name="VALIDATE_TAIL_LOSS", value=VALIDATE_TAIL_LOSS)
tf.add_to_collection(name="VALIDATE_TAIL_HIT", value=VALIDATE_TAIL_HIT)
tf.add_to_collection(name="TEST_FULL_COUNT", value=TEST_FULL_COUNT)
tf.add_to_collection(name="TEST_FULL_LOSS", value=TEST_FULL_LOSS)
tf.add_to_collection(name="TEST_FULL_HIT", value=TEST_FULL_HIT)
tf.add_to_collection(name="TEST_TAIL_COUNT", value=TEST_TAIL_COUNT)
tf.add_to_collection(name="TEST_TAIL_LOSS", value=TEST_TAIL_LOSS)
tf.add_to_collection(name="TEST_TAIL_HIT", value=TEST_TAIL_HIT)
tf.add_to_collection(name="PREDICT", value=PREDICT)

SAVER = tf.train.Saver()
SAVER.export_meta_graph(model_graph_path)

with tf.Session() as SESS:
    SESS.run(tf.global_variables_initializer())
    SAVER.save(sess=SESS, save_path=initial_model_path, write_meta_graph=False, write_state=False)