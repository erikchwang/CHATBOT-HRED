import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *

SAVER = tf.train.import_meta_graph(model_graph_path)
TEST_FULL_COUNT = tf.get_collection("TEST_FULL_COUNT")[0]
TEST_FULL_LOSS = tf.get_collection("TEST_FULL_LOSS")[0]
TEST_FULL_HIT = tf.get_collection("TEST_FULL_HIT")[0]
TEST_TAIL_COUNT = tf.get_collection("TEST_TAIL_COUNT")[0]
TEST_TAIL_LOSS = tf.get_collection("TEST_TAIL_LOSS")[0]
TEST_TAIL_HIT = tf.get_collection("TEST_TAIL_HIT")[0]

with tf.Session() as SESS:
    SAVER.restore(sess=SESS, save_path=train_model_path)

    COORD = tf.train.Coordinator()
    THREADS = tf.train.start_queue_runners(sess=SESS, coord=COORD)

    full_ppl, full_err, tail_ppl, tail_err = run_assess(
        SESS, TEST_FULL_COUNT, TEST_FULL_LOSS, TEST_FULL_HIT,
        TEST_TAIL_COUNT, TEST_TAIL_LOSS, TEST_TAIL_HIT, test_set_path
    )

    print "full dialogue ppl on test set is", full_ppl
    print "full dialogue err on test set is", full_err
    print "tail sentence ppl on test set is", tail_ppl
    print "tail sentence err on test set is", tail_err

    COORD.request_stop()
    COORD.join(THREADS)