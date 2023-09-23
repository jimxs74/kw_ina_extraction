import pandas as pd
import os
import sys

#2. rutin2 membuat syspath ke root utk aktifkan __init__.py
repo_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
sys.path.append(repo_root)

#3. rutin3 Load the dataset
dataset_path = os.path.join(repo_root, "Github/kw_ina_extraction/notebooks/postager_nlp-id/dataset_ekstraksi_r29_pos_sm.xlsx")
df = pd.read_excel(dataset_path)
df_pos = df['pos_sentence_list']

from positionrank import PositionRank

pos = {'NN', 'VP', 'NP'}
# 1. create a TextRank extractor.
extractor = PositionRank()

import ast 

df_keyphrases = pd.DataFrame(columns=['key_1', 'val_1', 'key_2', 'val_2', 'key_3', 'val_3'])

for i in range(len(df_pos)):
    try:
        text = df_pos[i]
        print('Processing Text', i, end='..! ')
        text1 = ast.literal_eval(text)
        extractor.load_document(input=text1)
        keyphrase = extractor.candidate_weighting()

        # Separate the key and value elements into separate lists
        keys = [item[0] for item in keyphrase]
        values = [item[1] for item in keyphrase]

        # Create a dictionary with the keys and values
        data = {
            'key_1': [keys[0]],
            'val_1': [values[0]],
            'key_2': [keys[1]],
            'val_2': [values[1]],
            'key_3': [keys[2]],
            'val_3': [values[2]],
        }
        df_keyphrase = pd.DataFrame(data)
        print('Done')
    except Exception as e:
        print('Error on text', i)
        data = {
            'key_1': [' '],
            'val_1': [0],
            'key_2': [' '],
            'val_2': [0],
            'key_3': [' '],
            'val_3': [0],
        }
        df_keyphrase = pd.DataFrame(data)
    # Concatenate df_keyphrase with df_keyphrases
    df_keyphrases = pd.concat([df_keyphrases, df_keyphrase], ignore_index=True)