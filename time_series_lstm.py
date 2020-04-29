"""
TODO:
- get the more uptodate data
- check correlations
"""



import argparse
import numpy
import pandas
from matplotlib import pyplot

from sklearn . model_selection import train_test_split as skl_train_test_split
from tensorflow . keras import layers as tfk_layers
from tensorflow . keras import models as tfk_models
from tensorflow . keras import optimizers as tfk_optimizers
from tensorflow . keras import regularizers as tfk_regularizers


batch_size = 64
nb_epochs = 229
past_len = 33
nb_hidden_features = 11
predictions_len = 21
optimizer = tfk_optimizers . Adam (lr = 0.001)
loss_function = 'mae'
dropout_rate = 0.05
regularizer_amplitude = 0.002
regularizer_type = 'L1'

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
                  dropout_rate = 0.05,
                  regularizer = None) :
  lstm_model = tfk_models . Sequential (
                 [ tfk_layers . LSTM (nb_hidden_features,
                                      input_shape = (input_len, 1),
                                      dropout = dropout_rate),
                   tfk_layers . Dense (predictions_len, kernel_regularizer = regularizer) ])
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
  sequence_data = cases_data . iloc [ : , : -1 ]
  nb_sequences = len (sequence_data)
  sequences = nb_sequences * [ None, ]
  sequence_index = 0
  for row_index, row in sequence_data . iterrows () :
    sequences [sequence_index] = row . values
    first_nonzero = (sequences [sequence_index] != 0) . argmax ()
    sequences [sequence_index] = sequences [sequence_index] [ first_nonzero : ] . astype (float)
    sequence_index += 1
  return sequences


def compute_sequence_variations (sequences,
                                 max_ratio_value = 2.9) :
  min_ratio_value = 1. / max_ratio_value
  nb_sequences = len (sequences)
  variations = nb_sequences * [ None, ]
  sequence_index = 0
  for sequence in sequences :
    ratios = sequence [ 1 : ] / sequence [ : -1 ]
    if (ratios . shape [0] == 0) :
      continue
    # this includes numpy.inf
    ratios [ratios > max_ratio_value] = max_ratio_value
    ratios [ratios < min_ratio_value] = min_ratio_value
    ratios [numpy . isnan (ratios)] = 1.
    variations [sequence_index] = numpy . log (ratios)
    sequence_index += 1
  return variations [ : sequence_index ]


def prepare_data_for_the_model (sequences,
                                past_len,
                                predictions_len,
                                do_remove_zero_data = False) :
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


def process_args () :
  global nb_epochs, random_seed, nb_hidden_features
  global predictions_len, batch_size, past_len
  global dropout_rate, test_data_size, max_ratio_value
  parser = argparse . ArgumentParser ()
  parser . add_argument ('--nb_epochs', type = int, default = nb_epochs)
  parser . add_argument ('--random_seed', type = int, default = random_seed)
  parser . add_argument ('--nb_hidden_features', type = int, default = nb_hidden_features)
  parser . add_argument ('--predictions_len', type = int, default = predictions_len)
  parser . add_argument ('--batch_size', type = int, default = batch_size)
  parser . add_argument ('--past_len', type = int, default = past_len)
  parser . add_argument ('--dropout_rate', type = float, default = dropout_rate)
  parser . add_argument ('--test_data_size', type = float, default = test_data_size)
  parser . add_argument ('--max_ratio_value', type = float, default = max_ratio_value)
  #parser . add_argument ('--loss', type = int, default = )
  #parser . add_argument ('--optimizer', type = int, default = )
  args = parser . parse_args ()
  nb_epochs = args . nb_epochs
  random_seed = args . random_seed
  nb_hidden_features = args . nb_hidden_features
  predictions_len = args . predictions_len
  batch_size = args . batch_size
  past_len = args . past_len
  dropout_rate = args . dropout_rate
  test_data_size = args . test_data_size
  max_ratio_value = args . max_ratio_value
  # = args . 
  # = args . 

def convert_model_output_to_predictions (model_output, initial_value, output_index = 0) :
  # TODO: test
  variation_rates = numpy . exp (model_output [output_index, : ])
  prediction_len = variation_rates . shape [0]
  predictions = numpy . zeros (prediction_len)
  predictions [0] = initial_value * variation_rates [0]
  for t in range (1, prediction_len) :
    predictions [t] = predictions [t - 1] * variation_rates [t]
  return predictions

def plot_comparison_graphs (area_name) :
  area_data = cases_data [cases_data ['area'] == area_name]
  area_cases = extract_unmodified_sequences_from_first_case (area_data) [0]
  area_variations = compute_sequence_variations ([ area_cases ],
                                               max_ratio_value = max_ratio_value) [0]
  data_past = area_variations [ : - predictions_len ]
  model_output = lstm_model . predict (numpy . expand_dims (numpy . expand_dims (data_past [ - past_len : ], axis = -1), axis = 0))
  predictions = convert_model_output_to_predictions (model_output, area_cases [ - predictions_len - 1 ])
  nb_days = area_variations . shape [0]
  days = numpy . arange (nb_days)
  pyplot . plot (days, area_cases [ - nb_days : ])
  pyplot . plot (days [ - predictions_len : ], predictions)
  pyplot . legend ([f'{area_name} data', 'Predictions'])
  pyplot . show ()


if __name__ == '__main__' :
  process_args ()
  cases_data = load_the_data ('time_series_covid19_confirmed_global.csv', do_correct_known_typos = True)
  sequences = extract_unmodified_sequences_from_first_case (cases_data)
  variations = compute_sequence_variations (sequences, max_ratio_value = max_ratio_value)
  train_sequences, test_sequences = skl_train_test_split (variations,
                                                          test_size = test_data_size,
                                                          shuffle = True,
                                                          random_state = random_seed)
  train_data_past, train_data_future = prepare_data_for_the_model (train_sequences, past_len, predictions_len, do_remove_zero_data = False)
  test_data_past, test_data_future = prepare_data_for_the_model (test_sequences, past_len, predictions_len, do_remove_zero_data = False)
  regularizer = None
  if (regularizer_type == 'L1') :
    regularizer = tfk_regularizers . l1 (l = regularizer_amplitude)
  elif (regularizer_type == 'L2') :
    regularizer = tfk_regularizers . l2 (l = regularizer_amplitude)
  lstm_model = create_and_prepare_model (input_len = past_len,
                                         nb_hidden_features = nb_hidden_features,
                                         predictions_len = predictions_len,
                                         optimizer = optimizer,
                                         loss_function = loss_function,
                                         dropout_rate = dropout_rate,
                                         regularizer = regularizer)
  train_history = lstm_model . fit (train_data_past,
                                    train_data_future,
                                    batch_size = batch_size,
                                    epochs = nb_epochs,
                                    validation_data = (test_data_past, test_data_future))
  train_history = train_history . history
  #print (f'Train history : {train_history}')
  min_loss_epoch = numpy . argmin (train_history ['val_loss'])
  print (f'Min loss at {min_loss_epoch} : ' + str (train_history ['val_loss'] [min_loss_epoch]))
  pyplot . plot (train_history ['loss'])
  pyplot . plot (train_history ['val_loss'])
  pyplot . legend (['Train loss', 'Validation loss'])
  pyplot . title ('Loss')
  pyplot . show ()
  #eval_results = lstm_model . evaluate (test_data_past, test_data_future, batch_size = batch_size)
  #print (f'Evaluation results : {eval_results}')
  plot_comparison_graphs ('United Kingdom - (main)')
  plot_comparison_graphs ('Italy - (main)')
  plot_comparison_graphs ('France - (main)')
  
  



