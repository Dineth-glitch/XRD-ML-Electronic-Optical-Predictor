# XRD Feature Extraction (Materials Project)

This project extracts fixed-length X-ray diffraction (XRD) feature vectors from crystal structures using the Materials Project API.

## Features
- Fetches structures using `mp-api`
- Generates XRD patterns using `pymatgen`
- Converts variable-length patterns into fixed 8192-length vectors
- Saves features as `.npy` file for ML models

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Add your Materials Project API key in the script:
```python
API_KEY = "YOUR_MATERIALS_PROJECT_API_KEY"
```

2. Run:
```bash
python xrd_feature_extraction.py
```

## Output
- `xrd_features.npy` → NumPy array of shape (N, 8192)
- Failed material IDs printed in terminal

## Notes
- Large dataset may take time depending on API limits
- Consider batching for very large runs
