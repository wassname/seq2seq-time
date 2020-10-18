This project has multiple ways of documenting requirements

- environment.yaml - This is the manual requirements, use it to install a new test or dev environment
- environment.min.yaml - This is the minimum requirements, use it to install a new test or dev environment
- environment.max.yaml - This pins all conda packages, use for production or finding vunrebilities
- requirements.txt - For people or bots not using conda

```
# Install requirements
conda create --name seq2seq-time python=3.7 -f ./requirements/environment.yaml
conda activate seq2seq-time 
# Install this package in editable mode
python -m pip install -e .
# Install kernel
python -m ipykernel install --user --name seq2seq-time --display-name seq2seq-time 
```
