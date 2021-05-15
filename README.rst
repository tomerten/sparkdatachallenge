==================
sparkdatachallenge
==================



Sparkdata challenge for finding multiplicative pairs in a sorted array of decimal numbers that
are constructed from two arrays (A,B), one containing the integer part and one containing the decimal part
but as an integer.

The decimal numbers are then constructed as following:
C[i] = A[i] + B[i] / scale 

where the scale is a fixed number (here 1_000_000).

* Free software: MIT license
* Documentation: https://sparkdatachallenge.readthedocs.io.


Features
--------

* Brute force method that fails due to memory allocation for large arrays but only uses numpy vectorized functions
* Brute force method based on a double for-loop
* Math based method - optimized using mathematical properties of the inequalities and leveraging that the decimal number array C is sorted.
  
