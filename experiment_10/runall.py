import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

# List of notebooks to be executed
notebooks =("1_tr_phrase_countvectorizer.ipynb", 
            "2_tr_phrase_pos_pattern.ipynb",
            "3_tr_posfilter.ipynb", 
            "4_tr_posfilter_posisi.ipynb",
            "5_tr_weightscorephrases.ipynb", 
            "6_tr_tfidfscore.ipynb",
            "summary")

# Iterate over the list and execute each notebook
for notebook in notebooks:
    print(f'Executing {notebook}...')
    with open(notebook, 'r') as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': './'}})
        print(f'{notebook} executed.')
        
        # Convert to HTML (optional)
        html_exporter = HTMLExporter()
        html_data, resources = html_exporter.from_notebook_node(nb)
        with open(notebook.replace('.ipynb', '.html'), 'w') as f:
            f.write(html_data)
            print(f'{notebook} converted to HTML.')
