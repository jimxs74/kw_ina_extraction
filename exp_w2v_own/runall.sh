#!/bin/bash

# List of notebooks to be executed
notebooks=("12_TextRank_cooccur.ipynb" "13a_TextRank_w2v_lg_sim.ipynb" "13a_TextRank_w2v_sm_sim.ipynb" "13b_TextRank_w2v_lg_combined.ipynb" "13b_TextRank_w2v_sm_combined.ipynb")

# Iterate over the list and execute each notebook
for notebook in "${notebooks[@]}"
do
    echo "Executing $notebook..."
    jupyter nbconvert --to html --execute "$notebook"
    echo "$notebook executed and converted to HTML."
done
