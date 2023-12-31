#!/bin/bash

# List of notebooks to be executed
notebooks=("SE132a_adjusted_textrank.ipynb" 
            "SE132b_adjusted_textrank.ipynb" 
            "SE132c_adjusted_textrank.ipynb" 
            "SE132d_adjusted_textrank.ipynb" )

# Iterate over the list and execute each notebook
for notebook in "${notebooks[@]}"
do
    echo "Executing $notebook..."
    jupyter nbconvert --to html --execute "$notebook"
    echo "$notebook executed and converted to HTML."
done
