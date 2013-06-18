## Control library
This C++ library aims to provide similar functionality to that provided by
Matlab's control systems toolbox.

## Requirements

### C++ code
* cmake 2.8 or later
* Eigen 3.1 or later
* LAPACKE

## Instructions

### Building the library and running tests

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    $ make test

### Examples
Examples are available in the form of unit tests.  See source/tests/test_*.cc
for basic use.

