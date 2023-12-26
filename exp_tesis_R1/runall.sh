#!/bin/bash

# List of notebooks to be executed
notebooks=("SE3_tr_phrase_pos_cooc_similar.ipynb" 
            "SE4_tr_w2v.ipynb" 
            "SE5_tr_w2v_posfilter.ipynb" 
            "SE6_tr_w2v_posfilter_weight1.ipynb" )

# Iterate over the list and execute each notebook
for notebook in "${notebooks[@]}"
do
    echo "Executing $notebook..."
    jupyter nbconvert --to html --execute "$notebook"
    echo "$notebook executed and converted to HTML."
done
