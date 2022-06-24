
# AST-Probe - Codebase and data
**Paper.** AST-Probe: Recovering abstract syntax trees from hidden representations of pre-trained language models
**Link.** https://arxiv.org/abs/2206.11719

## Installation 🛠️
1. Clone the repository.
```sh
git clone https://github.com/martin-wey/ast-probe
cd ast-probe
```
2. Create a python3 virtual environment and install `requirements.txt`.
```sh
python3 -m venv <name_of_your_env>
source <name_of_your_env>/bin/activate
pip install -r requirements.txt
```
We run out experiments using Python 3.9.10. In the `requirements.txt`, we include the PyTorch and CUDA versions that we used to report the results of the paper.

3. Install all tree-sitter grammars:
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
4. [Optional] You may have to add the project directory to your Python path.
```sh
export PYTHONPATH="${PYTHONPATH}:~/ast-probe/"
```
5. [Optional] Execute all tests.
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
The script `dataset_generator.py` with the argument `--download` will download the CodeSearchNet dataset, filter code snippets, and extract 20,000 samples for training, 4,000 for testing and 2,000 for validation. The filtering criteria are the following:
* We filter out code snippets that have a length `> 512` after tokenization.
* We remove code snippets that cannot be parsed by tree-sitter.
* We remove code snippets containing syntax errors

2. Train the AST-Probe.
```sh
python src/main.py \
  --do_train \
  --run_name <folder_run_name> \
  --pretrained_model_name_or_path <hugging_face_model> \
  --model_type <model_type> 
  --lang <lang> \
  --layer <layer> \
  --rank <rank>
```
The script `main.py` is in charge of training the probe. The main arguments are the following:
*  `--do_train`: if you want to train a probe classifier.
*  `--run_name`: indicates the name of the folder where the log, model and results will be stored.
*  `--pretrained_model_name_or_path`: the pre-trained model's id in the HuggingFace Hub.
*e.g.*, `microsoft/codebert-base`, `roberta-base`, `Salesforce/codet5-base`, etc.
*  `--model_type`: the model architecture. Currently, we only support `roberta` or `t5`.
*  `--lang`: programming language. Currently, we only support `python`, `javascript` or `go`.
*  `--layer`: the layer of the transformer model to probe. Normally, it goes from 0 to 12. If the pre-trained models is `huggingface/CodeBERTa-small-v1`, then this argument should range between 0 and 6.
*  `--rank`: dimension of the syntactic subspace.

As a result of this script, a folder `runs/folder_run_name` will be generated. This folder contains three files:
*  `ìnfo.log`: log file.
*  `pytorch_model.bin`: the probing model serialized *i.e.*, the basis of the syntactic subspace, the vectors C and U.
*  `metrics.log`: a serialized dictionary that contains the training losses, the validation losses, the precision, recall, and F1 score on the test set. You can use `python -m pickle runs/folder_run_name/metrics.log` to check it.

Here is an example of the usage of this script:
```sh
python src/main.py \
  --do_train \
  --run_name codebert_python_5_128 \
  --pretrained_model_name_or_path microsoft/codebert-base \
  --model_type roberta \
  --lang python \ 
  --layer 5 \
  --rank 128
```
This command trains a 128-dimensional probe over the output embeddings of the 5th layer of CodeBERT using the Python dataset. After running this command, the folder `runs/codebert_python_5_128` is created.

## Replicating the experiments of the paper
To replicate the experiments included in the paper, we provide two scripts that run everything.
- `run_experiments_rq123.py`: to replicate the results of RQ1, RQ2 and RQ3.
- `run_experiments_rq4.py`: to replicate the results of RQ4.

You may have to change a few things in these two scripts such as the `CUDA_VISIBLE_DEVICE`, *i.e.*, GPU used with PyTorch. Besides, the script will generate all the results for all experiments in separated folders such as described in the previous section of this readme.

### You can cite our work if you find this repository or the paper useful.
```sh
@misc{hernandez-ast-probe-2022,
  doi = {10.48550/ARXIV.2206.11719},
  url = {https://arxiv.org/abs/2206.11719},
  author = {López, José Antonio Hernández and Weyssow, Martin and Cuadrado, Jesús Sánchez and Sahraoui, Houari},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), Programming Languages (cs.PL), Software Engineering (cs.SE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {AST-Probe: Recovering abstract syntax trees from hidden representations of pre-trained language models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```