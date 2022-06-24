# Codebase for the Ast-Probe

## Installation 🛠️

1. Clone the repository.

```sh
git clone https://github.com/xxxxx/AstProbing.git
cd AstProbing
```

2. Create a virtual environment and install `requirements.txt`.

```sh
python3 -m venv <env_ast_probe>
source env_ast_probe/bin/activate
pip install -r requirements.txt
```

Our experiments were run with Python 3.9.10. In the `requirements.txt` we include the PyTorch and CUDA versions 
that we have used to report the results of the paper.

3. Install all the tree-sitter grammars:

```sh
mkdir grammars
cd grammars
git clone https://github.com/tree-sitter/tree-sitter-python.git
git clone https://github.com/tree-sitter/tree-sitter-javascript.git
git clone https://github.com/tree-sitter/tree-sitter-go.git
git clone https://github.com/tree-sitter/tree-sitter-php.git
git clone https://github.com/tree-sitter/tree-sitter-ruby.git
git clone https://github.com/tree-sitter/tree-sitter-java.git
cd ..
python src/data/build_grammars.py
```

4. Add project directory to Python path.

```sh
export PYTHONPATH="${PYTHONPATH}:~/AstProbing/"
```

5. Execute all tests (optional).
 
```sh
python -m unittest discover
```

## Running the probe 🚀

1. Dataset generation.

```sh
python src/dataset_generator.py --download --lang python
python src/dataset_generator.py --lang javascript
python src/dataset_generator.py --lang go
```

The script `dataset_generator.py` will download the CodeSearchNet dataset, filter code snippets and sample 
20000 for training, 4000 for testing and 2000 for validation. The criteria for filtering is the following:

* The length of the code snippet has to be less than 512.
* The code snippet can be parsed by tree-sitter.
* The code snippet does not contain syntactical errors.

2. Train the AST-probe.

```sh
python src/main.py --do_train --run_name <folder_run_name>
                   --pretrained_model_name_or_path <hugging_face_model>
                   --model_type <model_type> --lang <lang>
                   --layer <layer> --rank <rank>
```

The script `main.py` is in charge of training the probe, and the main arguments are the following:

* `--do_train`: It indicates that you want to train a probe.
* `--run_name`: It indicates the name of the folder where the log, model and results will be stored.
* `--pretrained_model_name_or_path`: It indicates the model's id in the Hub. 
  E.g., `microsoft/codebert-base`, `roberta-base`, `Salesforce/codet5-base`, etc.
* `--model_type`: It indicates the model architecture. Currently, we only support `roberta` or `t5`. 
* `--lang`: It indicates the considered programming language. Currently, we only support `python`, `javascript` or `go`.
* `--layer`: It indicates the layer of the transformer model. Normally, it goes from 0 to 12. If the pre-trained models 
  is `huggingface/CodeBERTa-small-v1`, then this argument admits an integer in the interval 0-6.
* `--rank`: It indicates the dimension of the syntactic subspace.

As a result of this script, a folder `runs/folder_run_name` will be generated. This folder contains three files:
* `ìnfo.log`: A log file.
* `pytorch_model.bin`: The probing model serialized i.e., the basis of the syntactic subspace, the vectors C and U.
* `metrics.log`: A serialized dictionary that contains the training losses, the validation losses, the precision, recall, 
and F1 score of the test set. You can use `python -m pickle runs/folder_run_name/metrics.log` to check it.
  
One example could be:

```sh
python src/main.py --do_train --run_name codebert_python_5_128
                   --pretrained_model_name_or_path microsoft/codebert-base
                   --model_type roberta --lang python
                   --layer 5 --rank 128
```

This command trains a 128-dimensional probe over the output embeddings of the 5th layer of CodeBERT using the Python dataset.
After running this command, the folder `runs/codebert_python_5_128` is created.