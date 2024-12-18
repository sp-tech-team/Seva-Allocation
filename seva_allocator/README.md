# Graph RAG with LLama Index


## prerequisites
* install Python3
* consider running: `pip3 install --upgrade pip`
* install required pip packages (requirements file in parent directory)
  * `python3 -m pip install -r requirements.txt`
* Some form of jupyter notebook should be installed either from VScode environment or from pip
  * can use pip3 install jupyter or anaconda
* Make sure you have the file `data/past_participant_info.csv`.
* Make sure you have the file `data/input_participant_info.csv`.
* Make sure you have the file `data/vrf_data.csv`
    * [VRF Google Doc](https://docs.google.com/spreadsheets/d/1r2YDv79x8mYN38kMj3219y149B0_lgbNJ_-58NzO8jo/edit?gid=842370995#gid=842370995)

## Running the Pipeline
1. Set up the default environment by running...
    * `echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

2. Train the model / build the database by running...
```
python3 indexer.py --setup_training_data
```
This creates the Vector Database/Index. This will timestamp each DB for each run.

3. Run inference and make job assignments with...
```
python3 inference.py \
--input_participant_info_csv 'data/input_participant_info.csv' \
--vrf_data_csv 'data/vrf_data.csv' \
--results_dir 'results/' \
--vector_store_base_dir 'vector_store_versions/' \
--vector_version 'latest' \
--num_samples 100 \
--num_job_predictions 3
```

This will read the latest Vector Index and run inference on the input data

## File Structure After Running Full Pipeline

- seva_allocator/
    - data/
        - generated_training_data/
            - vrf_generic_train_data.csv
            - vrf_specific_train_data.csv
        - input_participant_info.csv
        - past_participant_info.csv
        - vrf_data.csv
    - vector_store_versions/
        - index_YYYYMMDD_HHMMSS/
            - vector store files
    - results/
        - results-YYYYMMDD_HHMMSS
            - eval_results.csv
            - results_info.txt