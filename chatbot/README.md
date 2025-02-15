# Hybrid Chatbot

This project implements a hybrid conversational pipeline that combines structured and unstructured data for more dynamic query answering.

## Overview

1. **Structured Data:** Stored in a SQLite database named `participants_structured.db` containing columns like age, gender, years_of_experience.  
2. **Unstructured Data:** Stored in `participants_unstructured.db` holding free-text fields like skills, education_specialization, and past_jobs.  
3. **Embedding & Vector Store:** The pipeline uses OpenAI embeddings to convert unstructured data into vectors, which are indexed in a FAISS store for efficient similarity-based lookups.  
4. **LLM-Based Column Identification:** For each user query, the system attempts to identify which columns (structured and/or unstructured) are relevant.  
5. **SQL Generation & Execution:** If structured data is relevant, an LLM-based SQL generator creates and executes a query on the structured database.  
6. **Semantic Entities Extraction & Vector Search:** For unstructured data, the LLM extracts entities, which the FAISS index uses to retrieve relevant records.  
7. **Combined Response:** The structured and unstructured results are merged, then passed to an LLM to produce a final, coherent answer.

## How to Run

1. **Install Requirements:**  
   Make sure you have all dependencies installed (e.g., via `pip3 install -r requirements.txt`).

2. **Set Up .env:**  
   Provide your OpenAI API key and other environment variables in a `.env` file if needed.

3. **Check Participant Raw Data**
   Make sure from root the file `data/input_participant_info_raw.csv` is available for the preprocessing pipeline to load and build the database.

3. **Launch Chatbot:**  
   From the project root, run:  
   ```
   python3 -m chatbot.hybrid_chatbot
   ```
   - Optionally, use `--config_file_json <path/to/config.json>` to override default settings.

4. **Interact:**  
   You can now type queries into the console. Type `"exit"` to quit.

## Notes

- **Data Files:** Check `participants_structured.db` and `participants_unstructured.db` for the actual participant records.  
- **Logging:** Conversations are saved to `chatbot_conversation_log.txt` automatically.  
- **Mock Data:** Toggle `"use_mock_data": true` in the config to switch to a testing environment.
