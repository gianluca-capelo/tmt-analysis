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
   HAND_ANALYSIS_CSV variable in `config.py`:
   ```bash
   python -m src.runner.hand_analysis.run_hand_analysis

## Metadata

Must define the following variables in `config.py`: METADATA_CSV.
This should be the path to the metadata CSV file.

## Dependencies

Must install the dependencies listed in `requirements.txt`. You can do this by running:

   ```bash
   pip install -r requirements.txt
   ```

Keep in mind that _pyxations_ and _neurotask_ are still in development, so to install the latest version, you should
directly install from the GitHub repository of each package, using the following command:

   ```bash
   pip install -e .
   ```
