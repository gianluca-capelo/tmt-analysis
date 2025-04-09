# Analysis Pipeline Usage

Follow these steps to run the analysis:

1. **Add Pyxations Data:**
    - Place your Pyxations data in the `data` folder.
    - Update the `RAW_DATA_DIR` variable in `config.py` to point to the correct path for your data.

2. **Process the Data:**
   Run the following command from the root directory to process the data before analysis:
   ```bash
   python -m src.data_processing.process

3. **Run the Analysis:**
   Run the following command from the root directory to execute the hand analysis. The results will be saved in
   ANALYSIS_PATH variable in `config.py`:
   ```bash
   python -m src.hand_analysis.loader.analysis_loader
   

## Metadata

Before running the project, make sure you define the following variable in `config.py`:

- **METADATA_CSV**: The full path to your metadata CSV file.

For example:

    METADATA_CSV = os.path.join(DATA_DIR, "metadata", "metadata.csv")

## Dependencies

To install all required dependencies, run:

    pip install -r requirements.txt

> **Note:** Both _pyxations_ and _neurotask_ are actively under development. To ensure you are using the latest version
> of these packages, install them directly from their GitHub repositories using:
> _**pip install -e**_ .
