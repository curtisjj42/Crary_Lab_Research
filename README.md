# Crary_Lab_Research
Researching the relationship between cognitive testing batteries and functional brain regions

## Environment setup

You can set up the environment with either Conda (recommended on Windows due to OpenSlide dependency) or plain pip.

### Option A — Conda (recommended on Windows)

1. Install Miniconda or Anaconda.
2. Create the environment from `environment.yml`:

   ```
   conda env create -f environment.yml
   conda activate crary-lab
   ```

This will install Python, scientific packages, PDF tools, Jupyter, and OpenSlide (binaries and Python bindings) via conda-forge.

### Option B — pip

1. Ensure you have Python 3.10+ and a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

2. Install dependencies:

   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

Note about OpenSlide on Windows: `openslide-python` requires the OpenSlide binaries to be present on your system. If using pip on Windows, install the OpenSlide runtime first (or prefer the Conda option above). See: https://openslide.org/download/

### Quick verification

Run the following to verify all packages import correctly:

```
python - << "PY"
import sys
pkgs = [
    ("numpy", "np"),
    ("pandas", "pd"),
    ("matplotlib.pyplot", "plt"),
    ("seaborn", "sns"),
    ("pdfplumber", None),
    ("openslide", None),
]
for mod, alias in pkgs:
    try:
        if alias:
            globals()[alias] = __import__(mod, fromlist=[alias])
        else:
            __import__(mod)
        print(f"OK  - {mod}")
    except Exception as e:
        print(f"FAIL- {mod}: {e}")
        sys.exit(1)
print("All good!")
PY
```

## Data files
Place required input files under `data-files/` (see existing examples in that folder). Update notebooks to point to the correct paths as needed.
