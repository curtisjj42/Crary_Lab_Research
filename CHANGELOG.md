# Research Changelog

## 2025-10-02

### Project development
- Met with Jannes to discuss potential research directions
  - developed launchpad plan to enact project
  - got access to CSA and Mt Sinai systems
- Developed project workflow
  - Shrishtee, Mike, John, Kurt are primary inclusion

### Building Code
- Installed OpenSlide-python to begin working with WSIs
- experimented with loading WSIs and altering read parameters
  - **Notebook**: `notebooks/training/OpenSlide_learning.ipynb`
  - Data: 3 WSIs from PART cohort
  - 


## 2025-10-03

### Training
- built GitHub repo workflow with research oriented file structuring
- added sample experiment jupyter notebooks to track naming conventions
- added integration between notebooks and changelog & research_tasks

### Sample Experiment
- Adding sample integration workflow for running experiments
  - **Notebook**: `2025-10-03_sample_notebook.ipynb`
  - Accuracy: XX.X% (±CI%)
  - Training time: XXmin on GPU/computing cluster
  - **Outputs**:
    - Model Saved: `models/sample_v1.pkl`
    - Figures: `results/2025-10-03_baseline/`
  - **Issue**: model overfitting after epoch x
  - **Next**: Try early stopping (see notebook cell X comments)

### Sample Data Exploration
- Explored feature correlations in `notebooks/exploratory/01_EDA_sample_notebook.ipynb`
  - **Key finding**: Features X and Y are highly correlated (metric)
  - **Decision**: Will remove feature Y in next run