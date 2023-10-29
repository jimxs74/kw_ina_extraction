#!/bin/bash

# List of notebooks to be executed
notebooks=("1_tr_phrase_countvectorizer.ipynb" "2_tr_phrase_pos_pattern.ipynb" "3_tr_posfilter.ipynb" "4_tr_posfilter_posisi.ipynb" "5_tr_weightscorephrases.ipynb" "6_tr_tfidfscore.ipynb" "summary.ipynb")

# Iterate over the list and execute each notebook
for notebook in "${notebooks[@]}"
do
    echo "Executing $notebook..."
    jupyter nbconvert --to html --execute "$notebook"
    echo "$notebook executed and converted to HTML."
done
