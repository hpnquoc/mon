
## Getting Started
### Prerequisite

|            | Requirement                                                                                                                                                                                                                                          |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS**     | [**Ubuntu 20.04 / 22.04**](https://ubuntu.com/download/desktop) (fully supports), `Windows 10` and `MacOS` (partially supports)                                                                                                                      |
| **Env**    | [**Python>=3.9.0**](https://www.python.org/), [**PyTorch>=1.11.0**](https://pytorch.org/get-started/locally/), [**cudatoolkit=11.3**](https://pytorch.org/get-started/locally/), with [**anaconda**](https://www.anaconda.com/products/distribution) |
| **Editor** | [**PyCharm**](https://www.jetbrains.com/pycharm/download)                                                                                                                                                                                            |

### Directory
```text
zero_adce            # Root directory
 |__ data            # Contains data
 |__ runs            # Run dir
 |__ weights         # Pretrained models weights
```

### Installation
```shell
cd <your-preferred-location>
git clone https://phlong3105@github.com:phlong3105/Zero-ADCE zeroadce
# Enter password: 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

cd zeroadce
mkdir -p data

git clone https://phlong3105@github.com/phlong3105/one
cd one/install
chmod +x install.sh
conda init bash
bash -i install.sh
cd ..
conda activate one
pip install --upgrade -e .
```

### Run
In Terminal:
```shell
cd <zeroadce-save-location>
conda activate one
python infer.py --source "data/test/01.mp4"
```
