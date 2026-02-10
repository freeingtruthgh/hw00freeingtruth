#!/bin/bash
echo "Experiment 1: default parameters"
python multiclass_impl.py -t "hw02"

echo "Experiment 2: default parameters"
python multiclass_impl.py -t "hw02"

echo "Experiment 3: default parameters"
python multiclass_impl.py -t "hw02"

echo "Experiment 4: default parameters"
python multiclass_impl.py -t "hw02"

echo "Experiment 5: default parameters"
python multiclass_impl.py -t "hw02"

echo "All experiments completed!"

echo "Starting evaluation"
python multiclass_eval.py -t "hw02"

echo "Evaulation completed!"
