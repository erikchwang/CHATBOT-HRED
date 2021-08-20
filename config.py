train_raw_path = "data/raw_data/train_raw.pkl"
validate_raw_path = "data/raw_data/validate_raw.pkl"
test_raw_path = "data/raw_data/test_raw.pkl"
word_summary_path = "data/raw_data/word_summary.pkl"

train_set_path = "data/data_set/train_set.tfr"
validate_set_path = "data/data_set/validate_set.tfr"
test_set_path = "data/data_set/test_set.tfr"
word_index_path = "data/data_set/word_index.pkl"
index_word_path = "data/data_set/index_word.pkl"

model_graph_path = "model/model_graph"
initial_model_path = "model/initial_model"
train_model_path = "model/train_model"
train_temp_path = "model/train_temp.pkl"

bos_word = "<s>"
eos_word = "</s>"

rand_mean = 0.0
rand_stddev = 0.01

grad_clip_norm = 1.0
online_beam_width = 10

minimum_record_size = 10
maximum_record_size = 1000

original_learning_rate = 1e-04
terminal_learning_rate = 1e-07

learning_rate_decay_lock = 2
learning_rate_decay_rate = 0.5

update_batch_size = 5
assess_batch_size = 1

vocabulary_size = 10003
embedding_size = 400

encoder_cell_size = 400
context_cell_size = 1200
decoder_cell_size = 400