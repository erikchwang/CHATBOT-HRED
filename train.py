import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
import datetime
import glob

SAVER = tf.train.import_meta_graph(model_graph_path)
LEARNING_RATE = tf.get_collection("LEARNING_RATE")[0]
UPDATE = tf.get_collection("UPDATE")[0]
VALIDATE_FULL_COUNT = tf.get_collection("VALIDATE_FULL_COUNT")[0]
VALIDATE_FULL_LOSS = tf.get_collection("VALIDATE_FULL_LOSS")[0]
VALIDATE_FULL_HIT = tf.get_collection("VALIDATE_FULL_HIT")[0]
VALIDATE_TAIL_COUNT = tf.get_collection("VALIDATE_TAIL_COUNT")[0]
VALIDATE_TAIL_LOSS = tf.get_collection("VALIDATE_TAIL_LOSS")[0]
VALIDATE_TAIL_HIT = tf.get_collection("VALIDATE_TAIL_HIT")[0]

with tf.Session() as SESS:
    if not glob.glob(train_model_path + "*"):
        SAVER.restore(sess=SESS, save_path=initial_model_path)
        learning_rate = original_learning_rate
        epoch_number = 0
        full_ppls = []
        full_errs = []
        tail_ppls = []
        tail_errs = []
    else:
        SAVER.restore(sess=SESS, save_path=train_model_path)
        learning_rate, epoch_number, full_ppls, full_errs, tail_ppls, tail_errs = load_data(train_temp_path)

    cancel_count = 0
    COORD = tf.train.Coordinator()
    THREADS = tf.train.start_queue_runners(sess=SESS, coord=COORD)

    while learning_rate > terminal_learning_rate:
        begin_time = datetime.datetime.now()
        run_update(SESS, UPDATE, LEARNING_RATE, learning_rate, train_set_path)
        end_time = datetime.datetime.now()

        full_ppl, full_err, tail_ppl, tail_err = run_assess(
            SESS, VALIDATE_FULL_COUNT, VALIDATE_FULL_LOSS, VALIDATE_FULL_HIT,
            VALIDATE_TAIL_COUNT, VALIDATE_TAIL_LOSS, VALIDATE_TAIL_HIT, validate_set_path
        )

        print "epoch", epoch_number + 1, "takes", (end_time - begin_time).total_seconds(), "seconds"
        print "full dialogue ppl on validate set is", full_ppl
        print "full dialogue err on validate set is", full_err
        print "tail sentence ppl on validate set is", tail_ppl
        print "tail sentence err on validate set is", tail_err

        if epoch_number == 0 or full_ppl < full_ppls[-1]:
            epoch_number += 1
            full_ppls.append(full_ppl)
            full_errs.append(full_err)
            tail_ppls.append(tail_ppl)
            tail_errs.append(tail_err)
            SAVER.save(sess=SESS, save_path=train_model_path, write_meta_graph=False, write_state=False)
            dump_data([learning_rate, epoch_number, full_ppls, full_errs, tail_ppls, tail_errs], train_temp_path)
            print "this epoch is done, model is saved \n"
        else:
            cancel_count += 1
            SAVER.restore(sess=SESS, save_path=train_model_path)
            print "this epoch is canceled, model is restored to the last saved version \n"

            if cancel_count >= learning_rate_decay_lock:
                cancel_count = 0
                learning_rate *= learning_rate_decay_rate
                dump_data([learning_rate, epoch_number, full_ppls, full_errs, tail_ppls, tail_errs], train_temp_path)
                print "learning rate is decayed to", learning_rate, "\n"

    COORD.request_stop()
    COORD.join(THREADS)