# DGA Detection

Create virtual environment
```bash
conda env create --file local_env.yml
```

Activate and deactivate environment
```bash
conda activate local_dga
conda deactivate
```
Add environment to Jupyter Notebook
```bash
python -m ipykernel install --user --name=local_dga
```

Install latest pytorch for local and cpu only
```bash
conda install pytorch torchvision cpuonly -c pytorch
```

Install necessary packages (On your VM)
```bash
pip install -r gpu_requirements.txt
```
