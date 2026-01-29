# Project Repository Structure

## Overview
This repository contains the implementation and experimental results for the medical AI project.  
Due to dataset constraints, the source code and the dataset are intentionally organized in separate locations.

---

## Folder Structure

### `Medical_project/`
This directory contains the **core components of the project**, including:
- Main implementation files
- Model training and inference code
- Evaluation scripts
- Experiment results and analysis

All primary functions and results referenced in this project are located within this folder.

---

### Dataset Location (Outside `Medical_project/`)
The dataset used in this project is stored **outside** the `Medical_project` directory.

**Reason:**
- The dataset relies on **image URLs hosted on GitHub**.
- Moving the dataset or changing its directory structure would break these URL references.
- Broken URLs would prevent images from being correctly loaded during training and evaluation.

For this reason, the dataset must remain in its original location.

---

## Important Notes
- Do **not** move the dataset directory unless all image URLs are updated accordingly.
- All scripts inside `Medical_project/` assume the current dataset layout.
- Ensure the repository structure is preserved before running any experiments.

---

## Getting Started
1. Clone the repository.
2. Confirm that the dataset directory remains outside `Medical_project/`.
3. Run the main scripts or notebooks inside `Medical_project/`.

---
