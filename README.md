# AUTOMATIC LABELING OF MOUNTAIN BIKING SENSORY DATA

This is the code and the dataset to our submission "AUTOMATIC LABELING OF MOUNTAIN BIKING SENSORY DATA" to ICASSP 2021.

### Abstract
```
Supervised machine learning (ML) and deep learning (DL) algorithms have shown great results on labeled data in various research fields over this past decade.
However, in niche scenarios, such as mountain sports, labeled data is scarce.
Therefore, in this work, we present and evaluate an open source pipeline to easily create an auto-labeled, 25Hz sensory data set for the sport of downhill mountain biking.
We test the resulting data set against the two labels **ground surface** and **track difficulty** across four ML and DL algorithms and four window sizes.
The best models reach an adjusted balanced accuracy score of 0.54 +/-0.08 on ground surface classification and 0.56 +-0.06 on track difficulty classification.
From the empirical results we find that the proposed pipeline is a valid and easy option for recording and auto-labeling mountain sports sensory data sets for the purpose of ML and DL tasks.
```

### Structure
The repository is structured as follows:
* 01_auto_labeling: This is the code for converting and auto-labeling .fit files
* 02_data_set: This includes the data set used for evaluation
* 03_evaluation: This is the code used for the evaluation process