
from time_series_lstm import *

import unittest

import numpy
import pandas


def create_bit_of_fake_data () :
  data_dict = {}
  data_dict ['Country'] = [ 'base', 'base','base',
                            'shorter', 'shorter',
                            'longer', 'longer', 'longer', 'longer', 'longer']
  data_dict ['Date'] = [ '2020-01-02T00:00:00Z', '2020-01-03T00:00:00Z', '2020-01-04T00:00:00Z',
                         '2020-01-03T00:00:00Z', '2020-01-04T00:00:00Z',
                         '2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z', '2020-01-03T00:00:00Z', '2020-01-04T00:00:00Z', '2020-01-05T00:00:00Z', ]
  data_dict ['Confirmed'] = [ 1, 2, 3,
                              0, 10,
                              0, 0, 10, 0, 1000 ]
  data_dict ['Deaths'] = [ 0, 0, 0,
                           0, 0,
                           0, 0, 0, 0, 0 ]
  data_dict ['Recovered'] = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
  return pandas . DataFrame (data_dict)


class TestModelLstm (unittest . TestCase) :

  def test_extract_unmodified_sequences_from_first_case (self) :
    cases_data = create_bit_of_fake_data ()
    seq = extract_unmodified_sequences_from_first_case (cases_data)
    self . assertEqual (len (seq), 3)
    self . assertTrue (numpy . allclose (seq [0], numpy.array([ 1., 2., 3. ])))
    self . assertTrue (numpy . allclose (seq [1], numpy.array([ 10., ])))
    self . assertTrue (numpy . allclose (seq [2], numpy.array([ 10., 0., 1000.])))

  def test_compute_sequence_variations (self) :
    sequences = [
        numpy . array ([ 1., 2., 3., 2., 2. ]),
        numpy . array ([ 3., 0., 0., 0., 1., 8., 8.1]),
        numpy . array ([ 2., ]),
        numpy . array ([1.,1.,1.,1.]),
        numpy . array ([ 3., 0.001, 0.0011, 0.0012 ]), ]
    variations = compute_sequence_variations (sequences,
                                              max_ratio_value = 2.)
    logmax = numpy . log (2.)
    expected_variations = [
        numpy . log ([ 2., 1.5, 2./3., 1. ]),
        numpy . array ([ -logmax, 0., 0., logmax, logmax, numpy.log(8.1/8.)]),
        numpy . zeros (3),
        numpy . array ([ -logmax, numpy.log(0.11/0.1), numpy.log(0.12/0.11) ]), ]
    self . assertEqual (len (variations), 4)
    self . assertTrue (numpy . allclose (variations [0], expected_variations [0]))
    self . assertTrue (numpy . allclose (variations [1], expected_variations [1]))
    self . assertTrue (numpy . allclose (variations [2], expected_variations [2]))
    self . assertTrue (numpy . allclose (variations [3], expected_variations [3]))


if __name__ == '__main__' :
  unittest . main ()

