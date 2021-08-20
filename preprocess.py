import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *

word_summary = load_data(word_summary_path)

word_index = {item[0]: item[1] for item in word_summary}
dump_data(word_index, word_index_path)

index_word = {item[1]: item[0] for item in word_summary}
dump_data(index_word, index_word_path)

train_raw = load_data(train_raw_path)
dump_raw_data(train_raw, train_set_path, 3, word_index[bos_word], word_index[eos_word])

validate_raw = load_data(validate_raw_path)
dump_raw_data(validate_raw, validate_set_path, 3, word_index[bos_word], word_index[eos_word])

test_raw = load_data(test_raw_path)
dump_raw_data(test_raw, test_set_path, 3, word_index[bos_word], word_index[eos_word])