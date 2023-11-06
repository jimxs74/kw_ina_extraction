#!/bin/bash

# List of notebooks to be executed
notebooks=(
    "14b1_TR_w2v_lg_posfilter.ipynb" 
    "14b2_TR_w2v_lg_posfilter.ipynb" 
    )

# Iterate over the list and execute each notebook
for notebook in "${notebooks[@]}"
do
    echo "Executing $notebook..."
    jupyter nbconvert --to html --execute "$notebook"
    echo "$notebook executed and converted to HTML."
done
