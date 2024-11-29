# Graph RAG with LLama Index


## prerequisites
* install Python3
* consider running: `pip3 install --upgrade pip`
* install required pip packages (requirements file in parent directory)
  * `python3 -m pip install -r requirements.txt`
* Some form of jupyter notebook should be installed either from VScode environment or from pip
  * can use pip3 install jupyter or anaconda
* Make sure you have the file `data/past_participant_info.csv`.
* Make sure you have file `data/vrf_data.csv`
    * [VRF Google Doc](https://docs.google.com/spreadsheets/d/1r2YDv79x8mYN38kMj3219y149B0_lgbNJ_-58NzO8jo/edit?gid=842370995#gid=842370995)

## Running the Pipeline
1. Run `prep_train_test_data.ipynb`
    * This prepares the training and eval data.
2. Run `python3 graph_rag_index.py --vrf_jobs_train_corpus_txt "data/vrf_jobs_train_corpus.txt" --vrf_depts_train_corpus_txt "data/vrf_depts_train_corpus.txt" --index_config_json "configs/index_config.json"`
    * This creates the Property Graph and Vector Database/Index. This will timestamp each DB.
3. Run `python3 graph_rag_test.py`
    * This will read the latest Graph Rag Index and evaluate the model on the generated test data.

## Generated File
* `data/eval_results.csv`
* `data/vrf_depts_train_corpus.txt`
* `data/vrf_jobs_train_corpus.txt`


