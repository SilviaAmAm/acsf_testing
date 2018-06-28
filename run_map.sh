#!/bin/bash

source activate py36

for SAMPLES in 500 1000 3000 5000 8000 
    do
    for I in 1 2 3 4 5 6 7 8 9 10
        do
            python Speed_test_map.py $SAMPLES
        done
    done
