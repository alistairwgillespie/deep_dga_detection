# Deep DGA Detection 

# Getting Started
---

## 1. Installation Process

Clone and install environment dependencies
```bash
git clone ...
conda env create --file local_env.yml
conda install pytorch torchvision cpuonly -c pytorch
```

Activate environment
```bash
conda activate
```

Download data
```bash
dvc pull
```

Generate data streams to FLASK API
```bash
# Ensure you are in the root directory of the repo and run
FLASK_ENV=development FLASK_APP=app.py flask run
# Then open up another terminal and run
python eventgen.py
```

OPTIONAL: Add environment to Jupyter Notebook
```bash
python -m ipykernel install --user --name=threat_science
```

## References
---

*DGA 2016, Andrew Waeva, accessed 26 February 2020, https://github.com/andrewaeva/DGA.*

*Detecting DGA domains with recurrent neural networks and side information, Curtin, Gardner, Grzonkowski, Kleymenov, and Mosquera. 2019, accessed 26 February 2020, https://arxiv.org/pdf/1810.02023.pdf.*