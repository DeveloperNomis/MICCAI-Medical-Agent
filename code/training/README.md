# Detector training utilities

This folder contains utility scripts used to prepare YOLO-style detector training data from segmentation-derived annotations.

The original medical images, segmentation masks, and trained detector checkpoints are not included in this repository.

Typical preparation order:

1. `create_dataset_structure.py`  
   Creates a patient-level train/validation/test split and YOLO directory structure.

2. `generate_labels.py`  
   Generates initial YOLO label files from QA metadata.

3. `generate_gray_yolo_IDs.py`  
   Creates a mapping between grayscale segmentation identifiers and YOLO class IDs.

4. `extend_trainingData.py`  
   Adds additional training images while preserving patient-level separation from validation/test patients.

5. `extend_labelsTraining.py`  
   Generates YOLO labels for the extended training set.

6. `extend_labelsTraining_ClassMapping.py`  
   Maps dataset-specific anatomical labels to the final detector ontology.

Before running the scripts, replace placeholder paths such as `/path/to/...` with local dataset paths.