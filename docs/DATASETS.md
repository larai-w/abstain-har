# Dataset notes

**Selected research input source:** the pinned 22-participant UCI HARTH distribution. See [the source and input protocol](HARTH_PROTOCOL.md) and [source manifest](../contracts/harth-v1.2-source.json). A streaming HARTH window adapter is implemented; research model experiments remain pending. The manifest records the earlier metadata-only source inspection, not the scope of subsequent adapter runs.

The UCI HAR official archive was downloaded on 2026-09-26 for documentation and metadata inspection. Research feature matrices and sensor recordings have not been loaded, trained on, or evaluated. Input-adapter tests use generated synthetic files only.

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

## UCI HAR distribution inspection (2026-09-26)

Source: [UCI dataset page](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and its [official download](https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip).

The current landing page displays **CC BY 4.0**. The downloaded archive's `UCI HAR Dataset/README.txt` also contains the sentence **“Any commercial use is prohibited.”** These descriptions differ. Their applicability has not been resolved here; do not infer unrestricted use from this repository or treat the CLI as a terms-acceptance mechanism. Research-data model experiments remain pending clarification of the applicable conditions. No permission request has been sent on anyone's behalf.

Recorded hashes:

| Artifact | SHA-256 |
| --- | --- |
| Official outer ZIP | `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031` |
| Nested `UCI HAR Dataset.zip` | `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0` |

The inspected release documentation describes version 1.0, 561 derived features, 50 Hz sampling, and 128-reading windows with 50% overlap. Feature metadata has 561 ordered positions and 477 unique source names; duplicate names must not collapse columns. The README required Latin-1 decoding during inspection. Metadata was inspected directly inside the archives; raw feature matrices were not extracted into the repository.

The archive and local inspection record remain in ignored `data/`. No research vectors, raw recordings, or archive files are published here. See [the input adapter](HAR_INPUT.md) for positional validation and synthetic reproduction.

## Source selection (2026-09-26)

The [current UCI donation policy](https://archive.ics.uci.edu/contribute/donation) describes CC BY 4.0 for donated datasets, but does not specifically explain how the UCI HAR archive's older restriction was superseded. No dataset-specific resolution was found in the reviewed primary sources. UCI HAR remains on hold for research experiments here.

| Source | Evidence and implementation tradeoff | Selection |
| --- | --- | --- |
| UCI HAR | Existing 561-column adapter; conflicting page/archive descriptions remain. | Retain synthetic layout checks; defer research use. |
| PAMAP2 | [UCI page](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) lists CC BY 4.0, nine participants, missing values, and three IMUs. Archive terms have not been inspected here. | Defer; fewer people for four distinct roles and more input harmonisation. |
| HARTH, UCI copy | Dataset-specific CC BY 4.0 declaration; downloaded ZIP has 22 participant CSVs and no conflicting terms document. Named sensor channels require a new adapter. | Selected; pin archive and split before row-level analysis. |

This selection prioritises explicit source evidence and participant-separated evaluation. It does not establish that any model transfers between sensor placements, populations, or datasets.
