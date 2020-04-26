"""
TODO:
- test
- x valid
- ensemble?
"""

import numpy
import pandas
from matplotlib import pyplot

from sklearn . model_selection import train_test_split as skl_train_test_split
from tensorflow . keras import layers as tfk_layers
from tensorflow . keras import models as tfk_models
from tensorflow . keras import optimizers as tfk_optimizers


batch_size = 64
nb_epochs = 1
past_len = 33
nb_hidden_features = 2
predictions_len = 21
optimizer = tfk_optimizers . Adam (lr = 0.0001)
loss_function = 'mae'
dropout_rate = 0.05

test_data_size = 0.2
random_seed = 10000000000006660000000000001 % 666

max_ratio_value = 3.1

"""
bug: in tensorflow.keras :(
https://github.com/tensorflow/tensorflow/issues/27120
(https://github.com/tensorflow/tensorflow/issues/34983)
def create_and_prepare_model (
                  input_len = 33,
                  nb_hidden_features = 9,
                  predictions_len = 21,
                  optimizer = 'adam',
                  loss_function = 'mae',
                  dropout_rate = 0.05) :
  lstm_model = tfk_models . Sequential (
                 [ tfk_layers . LSTM (nb_hidden_features,
                                      input_shape = (input_len, 1),
                                      dropout = dropout_rate),
                   tfk_layers . Dense (predictions_len, activation = 'linear') ])
  lstm_model . compile (optimizer = optimizer, loss = loss_function)
  return lstm_model
"""

def create_and_prepare_model (
                  input_len = 33,
                  nb_hidden_features = 9,
                  predictions_len = 21,
                  optimizer = 'adam',
                  loss_function = 'mae',
                  dropout_rate = 0.05) :
  lstm_model = tfk_models . Sequential (
                 [ tfk_layers . LSTM (nb_hidden_features,
                                      input_shape = (input_len, 1),
                                      dropout = dropout_rate),
                   tfk_layers . Dense (predictions_len) ])
  lstm_model . compile (optimizer = optimizer, loss = loss_function)
  return lstm_model




def load_the_data (data_file_name, do_correct_known_typos = False) :
  cases_data = pandas . read_csv (data_file_name)
  cases_data ['Province/State'] = cases_data [['Province/State']] . fillna ('(main)', inplace = False)
  if (do_correct_known_typos) :
    cases_data . loc [207, 'Country/Region'] = 'Taiwan'
  cases_data ['area'] = cases_data ['Country/Region'] . astype (str) + ' - ' + cases_data ['Province/State'] . astype (str) 
  cases_data . drop ([ 'Province/State', 'Country/Region', 'Lat', 'Long' ], axis = 'columns', inplace = True)
  return cases_data

def extract_unmodified_sequences_from_first_case (cases_data) :
  # TODO: test this function
  sequence_data = cases_data . iloc [ : , : -1 ]
  nb_sequences = len (sequence_data)
  sequences = nb_sequences * [ None, ]
  for row_index, row in sequence_data . iterrows () :
    sequences [row_index] = row . values
    first_nonzero = (sequences [row_index] != 0) . argmax ()
    sequences [row_index] = sequences [row_index] [ first_nonzero : ] . astype (float)
  return sequences


def compute_sequence_variations (sequences,
                                 max_ratio_value = 2.9) :
  # TODO: test this function
  nb_sequences = len (sequences)
  variations = nb_sequences * [ None, ]
  for sequence_index, sequence in enumerate (sequences) :
    ratios = sequence [ 1 : ] / sequence [ : -1 ]
    # this includes numpy.inf
    ratios [ratios > max_ratio_value] = max_ratio_value
    ratios [ratios < - max_ratio_value] = - max_ratio_value
    ratios [numpy . isnan (ratios)] = 1.
    variations [sequence_index] = numpy . log (ratios)
  return variations


def prepare_data_for_the_model (sequences,
                                past_len,
                                predictions_len,
                                do_remove_zero_data = True) :
  # TODO: test this function
  model_inputs = []
  model_expected_outputs = []
  sequence_bit_len = past_len + predictions_len
  for sequence in sequences :
    sequence_len = sequence . shape [0]
    if (sequence_len <= predictions_len) :
      # theres nothing to learn here
      continue
    if (sequence_bit_len > sequence_len) :
      if (do_remove_zero_data) :
        continue
      input_sequence = numpy . zeros ((past_len, ), dtype = float)
      padding_len = past_len - (sequence_len - predictions_len)
      input_sequence [ padding_len : ] = sequence [ : - predictions_len ]
      model_inputs . append (input_sequence)
      model_expected_outputs . append (sequence [ - predictions_len : ])
    else :
      if (do_remove_zero_data and numpy . any (sequence == 0.)) :
        continue
      sequence_bit_start = 0
      while (sequence_bit_start + sequence_bit_len < sequence_len) :
        model_inputs . append (sequence [ sequence_bit_start : sequence_bit_start + past_len ])
        model_expected_outputs . append (sequence [ sequence_bit_start + past_len : sequence_bit_start + sequence_bit_len ])
        sequence_bit_start += 1
  return ( numpy . expand_dims (numpy . asarray (model_inputs), axis = -1),
           numpy . expand_dims (numpy . asarray (model_expected_outputs), axis = -1))

cases_data = load_the_data ('time_series_covid19_confirmed_global.csv', do_correct_known_typos = True)
sequences = extract_unmodified_sequences_from_first_case (cases_data)
variations = compute_sequence_variations (sequences, max_ratio_value = max_ratio_value)

data_past, data_future = prepare_data_for_the_model (variations, past_len, predictions_len, do_remove_zero_data = False)

( train_data_past,
  test_data_past,
  train_data_future,
  test_data_future ) = skl_train_test_split (data_past,
                                             data_future,
                                             test_size = test_data_size,
                                             shuffle = True,
                                             random_state = random_seed)


lstm_model = create_and_prepare_model (input_len = past_len,
                                       nb_hidden_features = nb_hidden_features,
                                       predictions_len = predictions_len,
                                       optimizer = optimizer,
                                       loss_function = loss_function,
                                       dropout_rate = dropout_rate)


train_history = lstm_model . fit (train_data_past,
                                  train_data_future,
                                  batch_size = batch_size,
                                  epochs = nb_epochs,
                                  validation_data = (test_data_past, test_data_future))

print (f'Train history : {train_history.history}')

eval_results = lstm_model . evaluate (test_data_past, test_data_future, batch_size = batch_size)
print (f'Evaluation results : {eval_results}')


## data_future = lstm_model . predict (data_past)

