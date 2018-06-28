#!/bin/bash

source activate py36

for SAMPLES in 500 1000 3000 5000 8000 
    do
    for BATCHES in 10 100 500
        do
        for I in 1 2 3 4 5
            do
                python Speed_test_batch.py $BATCHES $SAMPLES
            done
        done
    done
