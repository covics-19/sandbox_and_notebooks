
from time_series_lstm import *

import unittest

import numpy
import pandas



class TestModelLstm (unittest . TestCase) :

  def test_extract_unmodified_sequences_from_first_case (self) :
    data = numpy . array ([[0.,1.,2.,6.,-1.],[3.,4.,5.,7.,-2.],[0.,0.,0.,8.,-3.],[0.,9.,0.,10.,-4.]])
    cases_data = pandas . DataFrame (data = data, columns = [ f'col_{i}' for i in range (data . shape [1]) ])
    seq = extract_unmodified_sequences_from_first_case (cases_data)
    self . assertEqual (len (seq), 4)
    self . assertTrue (numpy . allclose (seq [0], numpy.array([1.,2.,6.])))
    self . assertTrue (numpy . allclose (seq [1], numpy.array([3.,4.,5.,7.])))
    self . assertTrue (numpy . allclose (seq [2], numpy.array([8.,])))
    self . assertTrue (numpy . allclose (seq [3], numpy.array([9.,0.,10.])))

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

