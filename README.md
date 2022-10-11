# Count Von Count (CVC)

CVC is a language model trained on CDR3 T-cell receptor sequences, built using a lightly modified BERT architecture with tweaked pre-training objectives. The model creates meaningful embeddings that can be used for downstream tasks like classification.

## Installation

To install CVC:
1. Clone the GitHub repository and create its requisite conda environment as follows.
   Make sure you use a recent conda version, e.g. version=4.10 or above

```bash
conda env create -n my_env_name_py39 python=3.9 --file=environment.yml
```

2. Upload model folder into the project base dir. Place the downloaded folder in the root of the project.

3. To create embeddings: <br />
   a. open notebook: lab_notebooks/create_embeddings.ipynb <br />
   b. edit first cell with relative paths <br />
   c. run noteboook <br />


TODO(P0): add model download instructions
* Install gdown: pip install --upgrade --no-cache-dir gdown