import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import *
import operator
import random
import nltk

SAVER = tf.train.import_meta_graph(model_graph_path)
DIALOGUE = tf.get_collection("DIALOGUE")[0]
BOS_LOCATIONS = tf.get_collection("BOS_LOCATIONS")[0]
EOS_LOCATIONS = tf.get_collection("EOS_LOCATIONS")[0]
PREDICT = tf.get_collection("PREDICT")[0]
word_index = load_data(word_index_path)
index_word = load_data(index_word_path)

with tf.Session() as SESS:
    def is_generic(word):
        if word.isalnum() or word in [bos_word, eos_word]:
            return False
        else:
            return True


    def get_sentence():
        user_input = raw_input("Input: ")

        if len(user_input) == 0:
            sentence = []
        else:
            sentence = [word_index[item] for item in nltk.word_tokenize(user_input) if word_index.has_key(item)]
            sentence.insert(0, word_index[bos_word])
            sentence.append(word_index[eos_word])

        return sentence


    def get_locations(dialogue):
        bos_locations = []
        eos_locations = []

        for location, value in enumerate(dialogue):
            if value == word_index[bos_word]:
                bos_locations.append(location)

            if value == word_index[eos_word]:
                eos_locations.append(location)

        return bos_locations, eos_locations


    def get_response(dialogue, minimum_word_count):
        response_candidates = [([word_index[bos_word]], 0.0)]
        word_count = 0

        while True:
            response_candidates = sorted(response_candidates, key=operator.itemgetter(1))

            if len(response_candidates) > online_beam_width:
                response_candidates = response_candidates[:online_beam_width]

            for response, _ in response_candidates:
                if response[-1] == word_index[eos_word]:
                    return response

            new_response_candidates = []

            for response, _ in response_candidates:
                temp_dialogue = dialogue + response + [word_index[eos_word]]
                temp_bos_locations, temp_eos_locations = get_locations(temp_dialogue)

                predict = run_predict(
                    SESS, PREDICT, DIALOGUE, BOS_LOCATIONS, EOS_LOCATIONS,
                    temp_dialogue, temp_bos_locations, temp_eos_locations
                )

                if word_count < minimum_word_count:
                    word_candidates = sorted(
                        [
                            (index, value) for index, value in enumerate(predict)
                            if not index == word_index[bos_word] and not is_generic(index_word[index])
                               and not index == word_index[eos_word]
                        ],
                        key=operator.itemgetter(1)
                    )
                else:
                    word_candidates = sorted(
                        [
                            (index, value) for index, value in enumerate(predict)
                            if not index == word_index[bos_word] and not is_generic(index_word[index])
                        ],
                        key=operator.itemgetter(1)
                    )

                if len(word_candidates) > online_beam_width:
                    word_candidates = word_candidates[:online_beam_width]

                for index, value in word_candidates:
                    new_response_candidates.append((response + [index], value))

            response_candidates = new_response_candidates
            word_count += 1


    SAVER.restore(sess=SESS, save_path=train_model_path)
    dialogue = []

    while True:
        sentence = get_sentence()

        if len(sentence) == 0:
            break

        dialogue.extend(sentence)
        response = get_response(dialogue, random.randint(1, 10))
        dialogue.extend(response)

        for item in response[1:-1]:
            print index_word[item],

        print "\n"