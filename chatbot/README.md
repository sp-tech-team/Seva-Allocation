# Seva Chatbot

This pipeline implements a chatbot for talking with a database consisting of bot structured and unstructured data about seva participants to help operators fill ashram roles.

## Overview

**Data**
* **CSV Participant Data:** We build the database from csv/ Google Sheets data downloaded from Sadhaka.
   * `data/input_participant_info_raw.csv` is the raw uncleaned data from Sadhaka
   * `data/input_participant_info_cleaned.csv` is the raw data reformated into a normalized table with not gaps.
   *  `data/input_participant_info_cleaned_mock{i}.csv` are the set of mock datasets exclusively used for testing.
* **Participant Data:** Stored in a Postgres database hosted on Supabase containing columns of participant information relavent to seva roles. Participant data is split into two tables: structured and unstructured data tables
   * participants_structured_data
   * participants_unstructured_data

* **Mock Data:** Stored in the same database as the real participant database, but with different table name prefixes. These tables are meant for testing purposes alone. The data here is cleaner and easier to write test questions.
   * participants_mock1_structured_data
   * participants_mock2_unstructured_data

* **Tests:** This document contains a suite of tests consisting of questions and answer pairs for each dataset to be used to evaluate the pipelines performance. This document comes in either json format (cleaner to read locally) or csv format to sync with the canonical testing data hosted on Google sheets.
   * Test data is stored in `chatbot/test_data/...`.
   * While the testing library (pg_text_to_sql_test.py) accepts both csv or json. you can use the script 
   ```
   python3 -m chatbot/scripts/pg_text_to_sql_test_convert.py --test_file_csv chatbot/test_data/tests.csv --ouput_file_json chatbot/test_data/tests_converted.json
   ```
   or 
   ```
   python3 -m chatbot/scripts/pg_text_to_sql_test_convert.py --test_file_json chatbot/test_data/tests.json --ouput_file_csv chatbot/test_data/tests_converted.csv
   ```

* **Text To SQL Prompts:** (unimplmented) If we have many text to sql prompts we can store them in `chatbot/configs/text_to_sql_pgvector_prompt.json`

## How to Run

1. **Install Requirements:**  
   Make sure you have all dependencies installed (e.g., via `pip3 install -r requirements.txt`).

2. **Set Up .env:**  
   Create a `chatbot/.env` and `database/.env` files with the following environment variables...
   ```
   OPENAI_API_KEY=...
   GRADIO_APP_PASSWORD=...
   SUPABASE_USER=postgres.vcilbhukibobtcednvvs
   SUPABASE_HOST=aws-0-ap-southeast-1.pooler.supabase.com
   SUPABASE_PORT=6543
   SUPABASE_NAME=postgres
   SUPABASE_PASSWORD=...
   ```

3. **Participant Data (csv):**  
   Make sure these files are available
      * `data/input_participant_info_raw.csv`
      * `data/input_participant_info_cleaned.csv`
      * `data/input_participant_info_cleaned_mock/2.csv`
   If the cleaned data is not available or you have new raw data from Sadhaka you can generate a cleaned file with...
   ```
   python3 -m database.participant_data --participant_info_raw_csv data/input_participant_info_raw.csv --output_participant_info_cleaned_csv data/input_participant_info_cleaned.csv
   ```
   

4. **Create Postgres Database**  
   Build the Postgres database. For the code it does not matter if the database is hosted locally or in the cloud. For now we hosting PG on Supabase. 
   * Setup a Supabase project (Database) (remember to save the password when creating). You can get the remaining Postgres Supabase parameters for you .env file from the connect button on the project home page. Use parameters from the `Transaction Pooler` section.
   * Install pgvector extension: On the left tab go to Database =>  Extensions => "Vector" toggle to install.  
   If using PSQL...
   ```
   CREATE DATABASE participants_mock1;
   \c participants_test
   CREATE EXTENSION vector;
   ```
   * Define the Database schema and upload the cleaned Sadhaka Data to Postgres database with the following script...
   ```
   python3 -m database.participant_pg_database --create_db --input_file_csv data/input_participant_info_cleaned_mock2.csv --table_base_name participants_mock2
   ```
   Note this script will also give some print outs simply testing the different interfaces of the database library. Also note --table_base_name is the prefix to the group of tables that pertain to one dataset in the database. For now we have 'participants', 'participants_mock1' and 'participants_mock2'. If you don't set --create_db it with simply load the existing database and print out some basic information  testing the python database inferface.

5. **Test Text To SQL Pipeline:**  
   * Make sure a test config file is available ex in `chatbot/test_data/tests.csv` as described in the Tests data overview section.
   ```
   python3 -m chatbot.pg_text_to_sql_test --test_file_json chatbot/test_data/tests_converted.json --results_file_json chatbot/test_results/eval_results.json
   ```  
   * Note you can also use `--run_tests_filter 'Basic Question Answer Pairs - Mock 1' 'Basic Question Answer Pairs - Mock 2' 'Basic Question Answer Pairs - Real 1'` to only run the listed tests you provide. Otherwise all tests will run.
   * You can also run the `pg_text_to_sql` pipeline more directly for a sanity check with `python3 -m chatbot.pg_text_to_sql --table_base_name participants_mock2`

6. **Run Commandline Chatbot:**  
The commandline chat bot lets you continuosly ask text to sql questions about the participant database. Note that currently there is no conversational aspect to it now. You can only ask text to SQL questions. We will make it more generically conversational in the future, using agent features.
   * You can run the following to start the chatbot...
   ```
   python3 -m chatbot.pg_chatbot --table_base_name participants_mock2
   ```
   * Your conversation will be logged to the file `chatbot/chatbot_conversation_log.txt`

7. **Run Gradio UI Chatbot:**  
The Gradio UI Chatbot is the same as the commandline chatbot but you will gave a pretty UI which will track your chat conversation history and display the SQL responses in markdown pretty table formate. Note this will crash after long conversations as gradio doesn't scale well easily. There will also be a password to login to the chatbot which you should define in the environment variables (.env file)
   * To run the Gradio UI Chatbot run...
   ```
   python3 -m chatbot.app --table_base_name participants_mock2
   ```
   * This will give you a hosting address you can put in your web browser: `http://127.0.0.1:7860` or `http://localhost:7860`


**Notes**  
* For Windows replace all `python3` with `python`

## Past Experiments in Git History

* Load all data into LLM context window:
   * data was loaded in both csv format as raw data and as sentences describing each participant line by line.
   * Both of these approaches work poorly for structured data but have been improving as models improve, but they will never be 100% reliable like SQL as the need to fully internallize the data and not hallucinate any copying mistakes. 
   * This approach should be repeaded every once in a while.
   * Its easy to try this wil chat gpt just uploading the data as a file. We should also have a version of this that uses the API and goes through our testing suite

* RAG and Graph RAG Chatbot
   * This approach involves making each of the participants data into sentences. Ex. Participant 21342 is 25 years old with 3 yoe in software dev. Each sentence is embededded and vector search is done for the user query. The question and vector response are supplied to the chat llm as context to answer the question.
   * Graph RAG is the same but in addition to vector search a graph search is done on a graph database. The graph database we tested was automically made by LlamaIndex/ Langchain on the participant data laid out into sentences. This creates a low quality graph though since these algos are expecting real english narratives not the same structured data thats repeated over and over in the same sentence. Instead in the future we should consider building a graph database with cipher and neo4j manually with custom knowledge of the relationships in the data. Also not that graph search can either be SQL like (Cipher or GraphQL) or vector search based where each node has an embedding.
* Text to SQL and Vector search invidually
   * This pipeline is very similar to our current pipeline except the vector search and structured sql search are generated separately and unaware of each other.
   * We used SQLite for the structured data and FAISS for the vector database.
   * [Design Doc](https://docs.google.com/document/d/1vc2ZzrDX2W5rHbugij7mADgyCfUiXARFZc1paDxHyYQ/edit?tab=t.0#heading=h.4415s17cs6zn)
   * This issue with this pipeline was the separation of information between structured and unstructured querying.
* Unfinished preprocessing Experiments
   * Clustering unstructured embeddings to make unstructured data more structured.
   * Using LLM to classify/group unstructured data into a more structured format.
* Future: 
   * Agentic chatbot: instead of just Text to sql a conversational main agent should schedule sub agents to do tasks like column analysis, text to sql and final response formatting. Sub agents that are complex like Text to sql will even have their own sub agents and tools.
   * Graph Database: hard code with Cypher a Relationship and Property graph of our data that makes fluid connections between structured and unstructured data. This can be used by the agentic bots too!
   * Vanna: a service that can help with text to sql by learning from examples and any sort of document based context to make text to sql better and more specific to specific rules our unusual text to sql will have
   * Hosting the chatbot somehwere somewhat stable. This is more important when the chatbot is relavent for more than just SP Seva assignment. It could be whole ashram level chatbot that needs to be able to scale.
