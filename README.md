# AUTOMATIC LABELING OF MOUNTAIN BIKING SENSORY DATA

This is the code and the dataset to our submission "AUTOMATIC LABELING OF MOUNTAIN BIKING SENSORY DATA" to EUSIPCO 2021.

### Abstract
```
Supervised machine learning (ML) and deep learning (DL) algorithms have shown great results on labeled data in various research fields over this past decade.
However, in niche scenarios, such as mountain sports, labeled data is scarce.
Therefore, in this work, we present and evaluate an open source pipeline to easily create large, auto-labeled, 25Hz sensory data sets for outdoor sports.
Utilizing our pipeline, we were able to automatically annotate recordings of smart wearable sensor data of multiple mountain bike riders with labels for ground surface and track difficulty.
On this data set, we trained four ML and DL classification algorithms on four window sizes, reaching adjusted balanced accuracy scores of 0.54 +/-0.08 (ground surface) and 0.56 +\-0.06$ (track difficulty).
From the empirical results we find that the proposed pipeline is a valid and easy option for quickly recording and auto-labeling mountain sports sensory data sets for the purpose of ML and DL tasks.
```

### Structure
The repository is structured as follows:
* 01_auto_labeling: This is the code for converting and auto-labeling .fit files
* 02_data_set: This includes the data set used for evaluation
* 03_evaluation: This is the code used for the evaluation process
