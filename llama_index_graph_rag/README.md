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
1. Set up the default environment by running...
    * `python3 graph_rag_setup_env.py --openai-key <add key here>`
    * All files created from here are user editable. Example you can edit `configs/index_config.json` to write a new schema for building the property graph
2. Train the model / build the database by running...
```
python3 graph_rag_index.py \
--vrf_jobs_train_corpus_txt "data/generated_training_data/vrf_jobs_train_corpus.txt" \
--vrf_depts_train_corpus_txt "data/generated_training_data/vrf_depts_train_corpus.txt" \
--index_config_json "configs/index_config.json"
```
This creates the Property Graph and Vector Database/Index. This will timestamp each DB.

3. Test the Graph RAG model by running...
```
python3 graph_rag_test.py \
--past_participant_info_csv 'data/past_participant_info.csv' \
--vrf_data_csv 'data/vrf_data.csv' \
--prompt_config_json 'configs/prompt_config.json' \
--results_dir 'results/' \
--results_info_file_name  'results_info.txt' \
--pg_store_dir 'pg_store_versions/' \
--vector_store_dir 'vector_store_versions/' \
--num_eval_samples 10 \
--inference_batch_size 10 \
--num_job_predictions 2
```

This will read the latest Graph Rag Index and evaluate the model on the generated test data.

## File Structure After Running Full Pipeline

- llama_index_graph_rag/
    - configs/
        - index_config.json
        - prompt_complete.txt
        - prompt_config.json
    - data/
        - generated_training_data/
            - vrf_jobs_train_corpus.txt
            - vrf_depts_train_corpus.txt
        - past_participant_info.csv
        - vrf_data.csv
    - pg_store_versions/
        - index_YYYYMMDD_HHMMSS/
            - graph store files
            - kg.html
    - vector_store_versions/
        - index_YYYYMMDD_HHMMSS/
            - vector store files
    - results/
        - results-YYYYMMDD_HHMMSS
            - eval_results.csv
            - results_info.txt


## Config Examples


**Filename:** `configs/index_config.json`

```
{
    "property_graph_schema_extractor": {
        "entities": ["JOB", "SKILL", "DEPARTMENT"],
        "relations": ["WORKS_WITH", "RELATED_TO", "SIMILAR_TO", "USED_BY", "IS_IN", "HAS"],
        "schema": {
            "JOB": ["RELATED_TO", "SIMILAR_TO", "WORKS_WITH", "IS_IN"],
            "SKILL": ["RELATED_TO", "USED_BY"],
            "DEPARTMENT": ["HAS", "WORKS_WITH"]
        }
    }
}
```

**Filename:** `configs/prompt_config.json`

```
{
    "prompt_mode": "COMPLETE",
    "prompt_complete_file": "configs/prompt_complete.txt"
}
```


**Filename:** `configs/prompt_complete.txt`

```
For a list of participants assign each participant a list of relevant jobs
from the provided list of jobs in the provided context only,
the list must be ranked by the strength of relevance to their skills and experience.
This assignment labeling task is called "Job Title Rank i"

Each participant should also get a label indicating their work skill level called "Skill Label".
Please assign the labels from this list[SKILLED-TECHNICAL, SKILLED-NON-TECHNICAL, NON-SKILLED]

The response must be formatted exactly as follows
"{Participant Id}/-/{Job Title Rank 1},{Job Title Rank 2},{Skill Label}"

Here is a list of rules to follow when generating the formatted response.
- Do not repeat job recommendations for the same person.
- Only assign jobs from the provided context.
- Do not add any other text to the response other than what's exactly described in the format string.
- Do not skip any of the response labels for a single participant.
- Do not put "Skill Label" labels one of the "Job Title" slots.
- The response for each participant must be on a new line.

Here is an example list of participants and the information we would have on each of them…
Participant 908902 has skills: IT / IT - Others. Python, C++, Machine Learning. specifically computer skills: data science, machine learning, python. The participant worked with designation: Software Engineer - Data and has a Bachelor of Science (B.Sc) education specialized in Computer Science
 Participant 909755 has skills: Soft Skills / General / Lawyer and specifically computer skills: Basic knowledge of M.S. Word, M.S. Excel, M.S. Powerpoint, Evernote, Adobe Photoshop, operating and navigating Manupatra.com (legal research platform). The participant worked with designation: Senior advocate and has a Bachelor of Law (B.L. / LLB) education specialized in Constitution of India
 Participant 972477 has skills: Basic Computer Skills / Basic Computer (MS Office and Email) Skills, Worked as Social Media Marketing Manager handling SG social media pages - YT, Twitter, Instagram and Facebook and specifically computer skills: Basic computer skills. The participant worked with designation: Social Media Marketing Manager and has a Master of Business Administration (M.B.A.) education specialized in Commerce

Here is an example of what your response would look like for these participants
908902/-/Data Scientist,Developer,SKILLED-TECHNICAL
909755/-/Lawyer,Administrative Activities (Back Office),SKILLED-NON-TECHNICAL
972477/-/Social media Manager,Content Strategist,NON-SKILLED

Here are the listed participants with their relevant metadata:
```



