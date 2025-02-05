# Seva Batch Allocator


## prerequisites
* install Python3
* consider running: `pip3 install --upgrade pip`
* `cd batch_allocator`
* setup a virtual environment `python3 -m venv venv`
* Add venv dir to .gitignore `batch_allocator/venv/`
* Activate / deactivate venv
    * `source myenv/bin/activate`
    * `deactivate`
* Update venv pip: `pip install --upgrade pip setuptools wheel`
* install required pip packages (requirements file in parent directory)
  * `python3 -m pip install -r requirements.txt`
* Some form of jupyter notebook should be installed either from VScode environment or from pip
    * can use pip3 install jupyter or anaconda
* Setup a pinecone and openai api account and get api keys


## Running the Pipeline
1. Set up the default environment by running...
    * ```
        echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
        echo "PINECONE_API_KEY=your_pinecone_api_key_here" >> .env
      ```

2. Train the model / build the database by running...
```
python3 vrf_indexer.py --pinecone_index_name vrf-vectors
```
This creates the Vector Database/Index. This will timestamp each DB for each run.

3. Run inference and make job assignments with...
```
python3 vrf_inference.py \
--input_participant_info_csv 'data/input_participant_info_raw.csv' \
--results_dir 'results/' \
--num_samples 100 \
--num_job_predictions 3 \
--pinecone_index_name vrf-vectors
```

This will read the latest Vector Index and run inference on the input data

## File Structure After Running Full Pipeline

- batch_allocator/
    - data/
        - generic_jobs.csv
        - input_participant_info_cleaned.csv
        - input_participant_info_raw.csv
        - vrf_data_cleaned.csv
        - vrf_data_raw.csv
    - results/
        - results-YYYYMMDD_HHMMSS
            - results.txt
    - .env
    - venv/