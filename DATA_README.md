# Data (manual step)

The ESA-ADB data is **not** redistributed in this repository.

1. Clone the official benchmark code next to this README (used for the official metric cross-check):
   ```
   git clone https://github.com/kplabs-pl/ESA-ADB
   ```
2. Download the ESA Anomaly Dataset from Zenodo: https://doi.org/10.5281/zenodo.12528696
   Only `ESA-Mission1.zip` (about 3.8 GB) and `ESA-Mission2.zip` (about 4.1 GB) are needed.
3. Unzip so that these folders exist:
   ```
   ESA-ADB/data/ESA-Mission1/   (channels/, labels.csv, anomaly_types.csv, channels.csv, ...)
   ESA-ADB/data/ESA-Mission2/
   ```
4. Build the 5-minute cache once: `bash reproduce.sh preprocess` (writes `cache/`, about 1 h).
