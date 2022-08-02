# AST-Probe - Installation

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
At this point, the repository is ready to run the scripts. Please, check the `README.md` to execute the scripts and replicate the experiments of the paper.
