# Dataset notes

No dataset has been downloaded, processed, or evaluated as part of this scaffold.

## UCI HAR

[Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) provides recordings from 30 participants, six activities, and a waist-mounted smartphone. Sensor sampling is 50 Hz. The release includes windowed sensor signals, 561 derived features, and participant identifiers.

The official training and test sets already separate participants. Internal model selection, calibration, and threshold selection also need participant separation; row-level random splits do not provide that protection.

Citation: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K

## PAMAP2

[PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) contains recordings from nine participants, with 100 Hz inertial sensors at the wrist, chest, and ankle. It includes missing values and a different activity inventory.

PAMAP2 is not a drop-in replacement for UCI HAR's feature vectors. Any cross-dataset experiment needs an explicit common input representation, units, sampling rate, windowing procedure, feature extraction, and activity mapping. Differences cannot be attributed to sensor placement alone.

Citation: Reiss, A. (2012). *PAMAP2 Physical Activity Monitoring*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NW2H

## Use and attribution

Both UCI landing pages displayed CC BY 4.0 on 2026-09-25. Before use, inspect the downloaded distribution's terms and reconcile any discrepancy. Record the source URL, retrieval date, checksum, attribution, and transformations. Do not commit raw recordings to this repository.

Results on these research datasets do not establish performance for older adults, clinical populations, or real-world care settings.
