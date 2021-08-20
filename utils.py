from config import *
import tensorflow as tf
import cPickle
import math


def load_data(file_name):
    with open(name=file_name, mode="rb") as file:
        data = cPickle.load(file)

    return data


def dump_data(data, file_name):
    with open(name=file_name, mode="wb") as file:
        cPickle.dump(obj=data, file=file, protocol=cPickle.HIGHEST_PROTOCOL)


def dump_raw_data(raw_data, file_name, turn_count, bos_index, eos_index):
    with tf.python_io.TFRecordWriter(file_name) as WRITER:
        for record in raw_data:
            if not minimum_record_size <= len(record) <= maximum_record_size:
                continue

            if not record.count(bos_index) == turn_count == record.count(eos_index):
                continue

            state = 0
            RECORD = tf.train.SequenceExample()
            DIALOGUE = RECORD.feature_lists.feature_list["DIALOGUE"]
            BOS_LOCATIONS = RECORD.feature_lists.feature_list["BOS_LOCATIONS"]
            EOS_LOCATIONS = RECORD.feature_lists.feature_list["EOS_LOCATIONS"]

            for location, value in enumerate(record):
                DIALOGUE.feature.add().int64_list.value.append(value)

                if state == 0 and value == bos_index:
                    BOS_LOCATIONS.feature.add().int64_list.value.append(location)
                    state = 1
                elif state == 1 and value == eos_index:
                    EOS_LOCATIONS.feature.add().int64_list.value.append(location)
                    state = 0
                elif state == 1 and value not in [bos_index, eos_index]:
                    pass
                else:
                    state = -1
                    break

            if state == 0:
                WRITER.write(RECORD.SerializeToString())
            else:
                continue


def get_record_count(file_name):
    record_count = 0

    for _ in tf.python_io.tf_record_iterator(file_name):
        record_count += 1

    return record_count


def load_record_batch(file_name, batch_size):
    _, ALL_RECORDS = tf.TFRecordReader().read_up_to(
        queue=tf.train.string_input_producer([file_name]),
        num_records=get_record_count(file_name)
    )

    _, PARSED_RECORD = tf.parse_single_sequence_example(
        serialized=tf.train.string_input_producer(ALL_RECORDS).dequeue(),
        sequence_features={
            "DIALOGUE": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
            "BOS_LOCATIONS": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64),
            "EOS_LOCATIONS": tf.FixedLenSequenceFeature(shape=[], dtype=tf.int64)
        }
    )

    DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH = tf.train.batch(
        tensors=[
            tf.cast(x=PARSED_RECORD["DIALOGUE"], dtype=tf.int32),
            tf.cast(x=PARSED_RECORD["BOS_LOCATIONS"], dtype=tf.int32),
            tf.cast(x=PARSED_RECORD["EOS_LOCATIONS"], dtype=tf.int32)
        ],
        batch_size=batch_size,
        dynamic_pad=True
    )

    return DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH


def get_encoder_batch(DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH):
    MAXIMUM_SIZE = tf.reduce_max(tf.subtract(x=EOS_LOCATIONS_BATCH[:, :-1], y=BOS_LOCATIONS_BATCH[:, :-1]))

    def SCAN_MATRIX(ARGUMENT):
        DIALOGUE = ARGUMENT[0]
        BOS_LOCATIONS = ARGUMENT[1]
        EOS_LOCATIONS = ARGUMENT[2]

        def SCAN_VECTOR(ARGUMENT):
            BOS_LOCATION = ARGUMENT[0]
            EOS_LOCATION = ARGUMENT[1]
            SENTENCE_SIZE = tf.subtract(x=EOS_LOCATION, y=BOS_LOCATION)
            PADDING_SIZE = tf.subtract(x=MAXIMUM_SIZE, y=SENTENCE_SIZE)
            SENTENCE_UNIT = tf.pad(tensor=DIALOGUE[BOS_LOCATION:EOS_LOCATION], paddings=[[0, PADDING_SIZE]])

            return [SENTENCE_SIZE, SENTENCE_UNIT]

        SENTENCE_SIZE_VECTOR, SENTENCE_UNIT_VECTOR = tf.map_fn(
            fn=SCAN_VECTOR,
            elems=[BOS_LOCATIONS, EOS_LOCATIONS],
            dtype=[tf.int32, tf.int32]
        )

        return [SENTENCE_SIZE_VECTOR, SENTENCE_UNIT_VECTOR]

    SENTENCE_SIZE_MATRIX, SENTENCE_UNIT_MATRIX = tf.map_fn(
        fn=SCAN_MATRIX,
        elems=[
            DIALOGUE_BATCH,
            tf.pad(tensor=BOS_LOCATIONS_BATCH[:, :-1], paddings=[[0, 0], [1, 0]]),
            tf.pad(tensor=EOS_LOCATIONS_BATCH[:, :-1], paddings=[[0, 0], [1, 0]])
        ],
        dtype=[tf.int32, tf.int32]
    )

    ENCODER_SENTENCE_SIZE_BATCH = tf.reshape(tensor=SENTENCE_SIZE_MATRIX, shape=[-1])
    ENCODER_SENTENCE_UNIT_BATCH = tf.reshape(tensor=SENTENCE_UNIT_MATRIX, shape=[-1, MAXIMUM_SIZE])

    return ENCODER_SENTENCE_SIZE_BATCH, ENCODER_SENTENCE_UNIT_BATCH


def get_decoder_batch(DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH):
    MAXIMUM_SIZE = tf.reduce_max(tf.subtract(x=EOS_LOCATIONS_BATCH, y=BOS_LOCATIONS_BATCH))

    def SCAN_MATRIX(ARGUMENT):
        DIALOGUE = ARGUMENT[0]
        BOS_LOCATIONS = ARGUMENT[1]
        EOS_LOCATIONS = ARGUMENT[2]

        def SCAN_VECTOR(ARGUMENT):
            BOS_LOCATION = ARGUMENT[0]
            EOS_LOCATION = ARGUMENT[1]
            SENTENCE_SIZE = tf.subtract(x=EOS_LOCATION, y=BOS_LOCATION)
            PADDING_SIZE = tf.subtract(x=MAXIMUM_SIZE, y=SENTENCE_SIZE)
            SENTENCE_UNIT = tf.pad(tensor=DIALOGUE[BOS_LOCATION:EOS_LOCATION], paddings=[[0, PADDING_SIZE]])
            SENTENCE_LABEL = tf.pad(tensor=DIALOGUE[BOS_LOCATION + 1:EOS_LOCATION + 1], paddings=[[0, PADDING_SIZE]])

            return [SENTENCE_SIZE, SENTENCE_UNIT, SENTENCE_LABEL]

        SENTENCE_SIZE_VECTOR, SENTENCE_UNIT_VECTOR, SENTENCE_LABEL_VECTOR = tf.map_fn(
            fn=SCAN_VECTOR,
            elems=[BOS_LOCATIONS, EOS_LOCATIONS],
            dtype=[tf.int32, tf.int32, tf.int32]
        )

        return [SENTENCE_SIZE_VECTOR, SENTENCE_UNIT_VECTOR, SENTENCE_LABEL_VECTOR]

    SENTENCE_SIZE_MATRIX, SENTENCE_UNIT_MATRIX, SENTENCE_LABEL_MATRIX = tf.map_fn(
        fn=SCAN_MATRIX,
        elems=[DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH],
        dtype=[tf.int32, tf.int32, tf.int32]
    )

    SENTENCE_MASK_MATRIX = tf.concat(
        values=[
            tf.zeros_like(tensor=SENTENCE_UNIT_MATRIX[:, :-1], dtype=tf.float32),
            tf.ones_like(tensor=SENTENCE_UNIT_MATRIX[:, -1:], dtype=tf.float32)
        ],
        axis=1
    )

    DECODER_SENTENCE_SIZE_BATCH = tf.reshape(tensor=SENTENCE_SIZE_MATRIX, shape=[-1])
    DECODER_SENTENCE_UNIT_BATCH = tf.reshape(tensor=SENTENCE_UNIT_MATRIX, shape=[-1, MAXIMUM_SIZE])
    DECODER_SENTENCE_LABEL_BATCH = tf.reshape(tensor=SENTENCE_LABEL_MATRIX, shape=[-1, MAXIMUM_SIZE])
    DECODER_SENTENCE_MASK_BATCH = tf.reshape(tensor=SENTENCE_MASK_MATRIX, shape=[-1, MAXIMUM_SIZE])

    return DECODER_SENTENCE_SIZE_BATCH, DECODER_SENTENCE_UNIT_BATCH, DECODER_SENTENCE_LABEL_BATCH, DECODER_SENTENCE_MASK_BATCH


class GRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, cell_size, is_scalar):
        self._cell_size = cell_size
        self._is_scalar = is_scalar

    @property
    def state_size(self):
        return self._cell_size

    @property
    def output_size(self):
        return self._cell_size

    def __call__(self, INPUT_BATCH, STATE_BATCH, SCOPE=None):
        RESET_GATE_BATCH = tf.contrib.layers.fully_connected(
            inputs=tf.concat(values=[STATE_BATCH, INPUT_BATCH], axis=1),
            num_outputs=1 if self._is_scalar else self._cell_size,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.orthogonal_initializer(),
            biases_initializer=tf.zeros_initializer()
        )

        UPDATE_GATE_BATCH = tf.contrib.layers.fully_connected(
            inputs=tf.concat(values=[STATE_BATCH, INPUT_BATCH], axis=1),
            num_outputs=1 if self._is_scalar else self._cell_size,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=tf.orthogonal_initializer(),
            biases_initializer=tf.zeros_initializer()
        )

        NEW_MEMORY_BATCH = tf.contrib.layers.fully_connected(
            inputs=tf.concat(values=[tf.multiply(x=STATE_BATCH, y=RESET_GATE_BATCH), INPUT_BATCH], axis=1),
            num_outputs=self._cell_size,
            activation_fn=tf.nn.tanh,
            weights_initializer=tf.orthogonal_initializer(),
            biases_initializer=tf.zeros_initializer()
        )

        NEW_STATE_BATCH = tf.add(
            x=tf.multiply(x=STATE_BATCH, y=tf.subtract(x=tf.ones_like(UPDATE_GATE_BATCH), y=UPDATE_GATE_BATCH)),
            y=tf.multiply(x=NEW_MEMORY_BATCH, y=UPDATE_GATE_BATCH)
        )

        OUTPUT_BATCH = NEW_STATE_BATCH

        return OUTPUT_BATCH, NEW_STATE_BATCH


def feed_forward(DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH):
    WORD_EMBEDDING = tf.get_variable(
        name="WORD_EMBEDDING",
        shape=[vocabulary_size, embedding_size],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(mean=rand_mean, stddev=rand_stddev)
    )

    ENCODER_SENTENCE_SIZE_BATCH, ENCODER_SENTENCE_UNIT_BATCH = get_encoder_batch(
        DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH
    )

    DECODER_SENTENCE_SIZE_BATCH, DECODER_SENTENCE_UNIT_BATCH, DECODER_SENTENCE_LABEL_BATCH, DECODER_SENTENCE_MASK_BATCH = get_decoder_batch(
        DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH
    )

    with tf.variable_scope("ENCODER"):
        _, ENCODER_CELL_STATES = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=GRUCell(encoder_cell_size, False),
            cell_bw=GRUCell(encoder_cell_size, False),
            inputs=tf.nn.embedding_lookup(params=WORD_EMBEDDING, ids=ENCODER_SENTENCE_UNIT_BATCH),
            sequence_length=ENCODER_SENTENCE_SIZE_BATCH,
            dtype=tf.float32
        )

        ENCODER_OUTPUTS = tf.reshape(
            tensor=tf.concat(values=ENCODER_CELL_STATES, axis=1),
            shape=[tf.shape(DIALOGUE_BATCH)[0], -1, encoder_cell_size * 2]
        )

    with tf.variable_scope("CONTEXT"):
        CONTEXT_CELL_OUTPUTS, _ = tf.nn.dynamic_rnn(
            cell=GRUCell(context_cell_size, False),
            inputs=ENCODER_OUTPUTS,
            dtype=tf.float32
        )

        CONTEXT_OUTPUTS = tf.contrib.layers.fully_connected(
            inputs=tf.reshape(tensor=CONTEXT_CELL_OUTPUTS, shape=[-1, context_cell_size]),
            num_outputs=decoder_cell_size,
            activation_fn=tf.nn.tanh,
            weights_initializer=tf.truncated_normal_initializer(mean=rand_mean, stddev=rand_stddev),
            biases_initializer=tf.zeros_initializer()
        )

    with tf.variable_scope("DECODER"):
        DECODER_CELL_OUTPUTS, _ = tf.nn.dynamic_rnn(
            cell=GRUCell(decoder_cell_size, False),
            inputs=tf.nn.embedding_lookup(params=WORD_EMBEDDING, ids=DECODER_SENTENCE_UNIT_BATCH),
            sequence_length=DECODER_SENTENCE_SIZE_BATCH,
            initial_state=CONTEXT_OUTPUTS,
            dtype=tf.float32
        )

        DECODER_OUTPUTS = tf.contrib.layers.bias_add(
            tf.add(
                x=tf.reshape(
                    tensor=tf.nn.embedding_lookup(params=WORD_EMBEDDING, ids=DECODER_SENTENCE_UNIT_BATCH),
                    shape=[-1, embedding_size]
                ),
                y=tf.contrib.layers.fully_connected(
                    inputs=tf.reshape(tensor=DECODER_CELL_OUTPUTS, shape=[-1, decoder_cell_size]),
                    num_outputs=embedding_size,
                    activation_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(mean=rand_mean, stddev=rand_stddev),
                    biases_initializer=None
                )
            )
        )

        DECODER_LOGITS = tf.contrib.layers.fully_connected(
            inputs=DECODER_OUTPUTS,
            num_outputs=vocabulary_size,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(mean=rand_mean, stddev=rand_stddev),
            biases_initializer=tf.zeros_initializer()
        )

    FULL_MASK = tf.sequence_mask(
        lengths=DECODER_SENTENCE_SIZE_BATCH,
        maxlen=tf.reduce_max(DECODER_SENTENCE_SIZE_BATCH),
        dtype=tf.float32
    )

    TAIL_MASK = tf.multiply(
        x=DECODER_SENTENCE_MASK_BATCH,
        y=tf.sequence_mask(
            lengths=DECODER_SENTENCE_SIZE_BATCH,
            maxlen=tf.reduce_max(DECODER_SENTENCE_SIZE_BATCH),
            dtype=tf.float32
        )
    )

    CLASS_SUCCESS = tf.reshape(
        tensor=tf.cast(
            x=tf.nn.in_top_k(
                predictions=DECODER_LOGITS,
                targets=tf.reshape(tensor=DECODER_SENTENCE_LABEL_BATCH, shape=[-1]),
                k=1
            ),
            dtype=tf.float32
        ),
        shape=tf.shape(DECODER_SENTENCE_LABEL_BATCH)
    )

    CROSS_ENTROPY = tf.reshape(
        tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(tensor=DECODER_SENTENCE_LABEL_BATCH, shape=[-1]),
            logits=DECODER_LOGITS
        ),
        shape=tf.shape(DECODER_SENTENCE_LABEL_BATCH)
    )

    OVERALL_ENTROPY = tf.reshape(
        tensor=tf.negative(tf.nn.log_softmax(DECODER_LOGITS)),
        shape=tf.concat(values=[tf.shape(DECODER_SENTENCE_LABEL_BATCH), [vocabulary_size]], axis=0)
    )

    return FULL_MASK, TAIL_MASK, CLASS_SUCCESS, CROSS_ENTROPY, OVERALL_ENTROPY


def model_update(LEARNING_RATE, file_name, is_reuse):
    DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH = load_record_batch(file_name, update_batch_size)

    with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=is_reuse):
        FULL_MASK, TAIL_MASK, CLASS_SUCCESS, CROSS_ENTROPY, OVERALL_ENTROPY = feed_forward(
            DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH
        )

        OBJECTIVE = tf.divide(
            x=tf.reduce_sum(tf.multiply(x=FULL_MASK, y=CROSS_ENTROPY)),
            y=tf.count_nonzero(input_tensor=FULL_MASK, dtype=tf.float32)
        )

        VARIABLES = tf.trainable_variables()
        GRADIENTS = tf.gradients(ys=OBJECTIVE, xs=VARIABLES)
        CLIPPED_GRADIENTS, _ = tf.clip_by_global_norm(t_list=GRADIENTS, clip_norm=grad_clip_norm)
        UPDATE = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(CLIPPED_GRADIENTS, VARIABLES))

    return UPDATE


def model_assess(file_name, is_reuse):
    DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH = load_record_batch(file_name, assess_batch_size)

    with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=is_reuse):
        FULL_MASK, TAIL_MASK, CLASS_SUCCESS, CROSS_ENTROPY, OVERALL_ENTROPY = feed_forward(
            DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH
        )

        FULL_COUNT = tf.count_nonzero(input_tensor=FULL_MASK, dtype=tf.float32)
        FULL_LOSS = tf.reduce_sum(tf.multiply(x=FULL_MASK, y=CROSS_ENTROPY))
        FULL_HIT = tf.reduce_sum(tf.multiply(x=FULL_MASK, y=CLASS_SUCCESS))
        TAIL_COUNT = tf.count_nonzero(input_tensor=TAIL_MASK, dtype=tf.float32)
        TAIL_LOSS = tf.reduce_sum(tf.multiply(x=TAIL_MASK, y=CROSS_ENTROPY))
        TAIL_HIT = tf.reduce_sum(tf.multiply(x=TAIL_MASK, y=CLASS_SUCCESS))

    return FULL_COUNT, FULL_LOSS, FULL_HIT, TAIL_COUNT, TAIL_LOSS, TAIL_HIT


def model_predict(DIALOGUE, BOS_LOCATIONS, EOS_LOCATIONS, is_reuse):
    DIALOGUE_BATCH = tf.expand_dims(input=DIALOGUE, axis=0)
    BOS_LOCATIONS_BATCH = tf.expand_dims(input=BOS_LOCATIONS, axis=0)
    EOS_LOCATIONS_BATCH = tf.expand_dims(input=EOS_LOCATIONS, axis=0)

    with tf.variable_scope(name_or_scope=tf.get_variable_scope(), reuse=is_reuse):
        FULL_MASK, TAIL_MASK, CLASS_SUCCESS, CROSS_ENTROPY, OVERALL_ENTROPY = feed_forward(
            DIALOGUE_BATCH, BOS_LOCATIONS_BATCH, EOS_LOCATIONS_BATCH
        )

        RESPONSE_SIZE = tf.count_nonzero(input_tensor=TAIL_MASK, dtype=tf.int32)

        PREDICT = tf.divide(
            x=tf.cond(
                pred=tf.equal(x=RESPONSE_SIZE, y=1),
                true_fn=lambda: OVERALL_ENTROPY[-1, 0],
                false_fn=lambda: tf.add(
                    x=OVERALL_ENTROPY[-1, RESPONSE_SIZE - 1],
                    y=tf.reduce_sum(CROSS_ENTROPY[-1, :RESPONSE_SIZE - 1])
                )
            ),
            y=tf.cast(x=RESPONSE_SIZE, dtype=tf.float32)
        )

    return PREDICT


def run_update(SESS, UPDATE, LEARNING_RATE, learning_rate, file_name):
    for _ in range(get_record_count(file_name) // update_batch_size):
        SESS.run(fetches=UPDATE, feed_dict={LEARNING_RATE: learning_rate})


def run_assess(SESS, FULL_COUNT, FULL_LOSS, FULL_HIT, TAIL_COUNT, TAIL_LOSS, TAIL_HIT, file_name):
    full_count_sum = full_loss_sum = full_hit_sum = tail_count_sum = tail_loss_sum = tail_hit_sum = 0.0

    for _ in range(get_record_count(file_name) // assess_batch_size):
        full_count_cur, full_loss_cur, full_hit_cur, tail_count_cur, tail_loss_cur, tail_hit_cur = SESS.run(
            [FULL_COUNT, FULL_LOSS, FULL_HIT, TAIL_COUNT, TAIL_LOSS, TAIL_HIT]
        )

        full_count_sum += full_count_cur
        full_loss_sum += full_loss_cur
        full_hit_sum += full_hit_cur
        tail_count_sum += tail_count_cur
        tail_loss_sum += tail_loss_cur
        tail_hit_sum += tail_hit_cur

    full_ppl = math.exp(full_loss_sum / full_count_sum)
    full_err = 1.0 - full_hit_sum / full_count_sum
    tail_ppl = math.exp(tail_loss_sum / tail_count_sum)
    tail_err = 1.0 - tail_hit_sum / tail_count_sum

    return full_ppl, full_err, tail_ppl, tail_err


def run_predict(SESS, PREDICT, DIALOGUE, BOS_LOCATIONS, EOS_LOCATIONS, dialogue, bos_locations, eos_locations):
    predict = SESS.run(
        fetches=PREDICT,
        feed_dict={DIALOGUE: dialogue, BOS_LOCATIONS: bos_locations, EOS_LOCATIONS: eos_locations}
    )

    return predict