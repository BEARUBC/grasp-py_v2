# grasp-py_v2

## Setup
*Requirements:*
- Python >= 3.8

Virtual environment
- If using conda:
    ```bash
    conda env create -f environment.yaml
    conda activate grasp_analytics_v2
    ```
- If using python venv:
    ```bash
    python3 -m venv grasp_env
    source grasp_env/bin/activate # Linux/macOS
    grasp_env\Scripts\activate # Windows
    pip install -r requirements.txt
    ```

Install Git LFS to download the model files
```bash
git lfs install
git lfs track "*.jpg"
git lfs track "*.txt"
```