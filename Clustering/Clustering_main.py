import pandas as pd
import numpy as np
import csv
import ast
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from collections import defaultdict
import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
print(os.getenv('OPENAI_API_KEY'))

messages = [ {"role": "system", "content":
              "You are a intelligent assistant."} ]

# Initialize OpenAI client
client = openai

# Configurable message templates
Split_tasks_Prompt_message = '''These are the Work Experience and the type of task that they do in their Job 
Figure specific tasks or skills from it and give it back as a comma separated string
Only the output and no other string is needed on the response
Here is the tasks: '''

# --- Step 1: Get and Clean Initial Data ---
def get_cleaned_tasks(input_file, tasks_column, sample_size=20):
    """
    Reads the input CSV and cleans the specified column.

    Args:
        input_file (str): Path to the CSV file.
        tasks_column (str): Column containing the tasks.
        sample_size (int): Number of tasks to process.
    
    Returns:
        pd.Series: Cleaned tasks.
    """
    df = pd.read_csv(input_file)
    tasks = df[tasks_column].dropna()  # Remove NaN
    tasks = tasks[tasks.str.strip() != ""]  # Remove blank values
    tasks = tasks.reset_index(drop=True)  # Reset index after cleaning
    print(f"Total cleaned tasks: {len(tasks)}")
    tasks_sample = tasks[:sample_size]
    print(f"Processing the first {len(tasks_sample)} tasks.")
    return tasks_sample

# --- Step 2: Get Task List from ChatGPT ---
def fetch_task_list(tasks, output_file, entities_file):
    """
    Uses OpenAI ChatGPT to extract task lists and saves results.

    Args:
        tasks (pd.Series): List of tasks to process.
        output_file (str): Path to save task responses.
        entities_file (str): Path to save entities.
    """
    output = []
    Entities = []

    for i, task in enumerate(tasks):
        print(f"Processing Task {i+1}: {task}")
        if task:
            messages = [{"role": "user", "content": f"{Split_tasks_Prompt_message}{task}"}]
            chat = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages  # gpt-4o-mini, gpt-3.5-turbo
            )
            reply = chat.choices[0].message.content
            entity_list = reply.split(',')
            Entities.append(entity_list)
            print(f"ChatGPT Response for Task {i+1}: {reply}")
            output.append({"Task": task, "Response": reply})

    # Save task responses
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["Task", "Response"])
        writer.writeheader()
        writer.writerows(output)

    # Save entities
    with open(entities_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Entities"])
        writer.writerow([str(Entities)])

    print(f"Responses saved to {output_file}")
    print(f"Entities saved to {entities_file}")

# --- Step 3: Create Nodes with Embeddings ---
def create_nodes_with_embeddings(input_file, output_file):
    """
    Creates nodes with embeddings and saves to a CSV.

    Args:
        input_file (str): Path to the CSV with tasks and responses.
        output_file (str): Path to save nodes with embeddings.
    """
    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding

    df_responses = pd.read_csv(input_file)
    nodes = []

    for idx, row in df_responses.iterrows():
        task = row["Task"]
        response = row["Response"]

        if pd.notna(response):
            entities = response.split(",")
            for entity_idx, entity in enumerate(entities):
                entity = entity.strip()
                metadata = {
                    "Task": task,
                    "Entity Name": f"Entity-{entity_idx + 1}",
                    "Entity Value": entity,
                }
                embedding = get_embedding(entity)
                node = {
                    "text": entity,
                    "metadata": metadata,
                    "embedding": embedding,
                }
                nodes.append(node)

    df_nodes = pd.DataFrame.from_records(
        [
            {
                "Entity": node["text"],
                "Task": node["metadata"]["Task"],
                "Entity Name": node["metadata"]["Entity Name"],
                "Entity Value": node["metadata"]["Entity Value"],
                "Embedding": node["embedding"],
            }
            for node in nodes
        ]
    )
    df_nodes.to_csv(output_file, index=False)
    print(f"Nodes saved to {output_file}")

# --- Step 4: Perform Clustering ---
def perform_hierarchical_clustering(input_file, output_clusters_file, dendrogram_file=None, distance_threshold=0.11):
    """
    Perform hierarchical clustering and cut the dendrogram at a given distance threshold.

    Args:
        input_file (str): Path to the CSV with nodes and embeddings.
        output_clusters_file (str): Path to save clustered entities.
        dendrogram_file (str): Path to save dendrogram image (optional).
        distance_threshold (float): Distance threshold for cutting clusters.
    """
    # Step 1: Load embeddings
    df_nodes = pd.read_csv(input_file)
    df_nodes = df_nodes[:112] #Select first few datas
    df_nodes["Embedding"] = df_nodes["Embedding"].apply(ast.literal_eval)  # Parse embedding strings
    embeddings_np = np.array(df_nodes["Embedding"].to_list())
    labels = df_nodes["Entity"].values

    print(f"Number of embeddings: {len(embeddings_np)}, Embedding dimensions: {embeddings_np.shape[1]}")

    # Step 2: Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    Z = linkage(embeddings_np, method="complete", metric="cosine", optimal_ordering=False)

    # Step 3: Plot dendrogram (if required)
    if dendrogram_file:
        print("Generating dendrogram...")
        plt.figure(figsize=(20, 10))
        dendrogram(
            Z,
            labels=labels,
            leaf_rotation=0,
            leaf_font_size=6,
            orientation="right",
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Index of Embeddings")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(dendrogram_file, format="png", dpi=300)
        plt.show()

    # Step 4: Cut the dendrogram at the specified distance threshold
    print(f"Cutting clusters at distance threshold: {distance_threshold}")
    cluster_labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Step 5: Group entities and tasks by cluster labels
    grouped_entities = defaultdict(list)
    grouped_tasks = defaultdict(set)  # Use a set to avoid duplicate tasks

    for i, label in enumerate(cluster_labels):
        entity = labels[i]
        grouped_entities[label].append(entity)
        
        # Retrieve tasks associated with the entity
        print('This is entity:', entity)
        tasks = df_nodes.loc[df_nodes["Entity"] == entity, "Task"].values
        
        if len(tasks) > 0:  # Ensure tasks are not empty
            grouped_tasks[label].update(tasks)  # Add tasks to the cluster (use set to avoid duplicates)

    # Include tasks in the cluster output
    clusters_data = [
        {
            "Cluster": cluster,
            "Entities": ", ".join(entities),
            "Tasks": ", ".join(grouped_tasks[cluster]) if cluster in grouped_tasks else ""
        }
        for cluster, entities in grouped_entities.items()
    ]

    df_clusters = pd.DataFrame(clusters_data)
    df_clusters.to_csv(output_clusters_file, index=False)
    print(f"Clusters saved to {output_clusters_file}")

    # Step 7: Summary of clusters
    for label, group in grouped_entities.items():
        print(f"Cluster {label}: {group}")

    # Print the total number of clusters
    total_clusters = len(grouped_entities)
    print(f"Total clusters formed: {total_clusters}")

    return grouped_tasks, grouped_entities, total_clusters

# --- Step 5: Name Clusters with GPT ---
def name_clusters_with_gpt(grouped_tasks, grouped_entities, client, cluster_names_file):
    """
    Name clusters using GPT by passing entity lists for each cluster.

    Args:
        grouped_entities (dict): Dictionary of clusters and their entities.
        client (object): OpenAI client instance for GPT interaction.
        cluster_names_file (str): Path to save cluster names.
    """
    cluster_names = {}
    for cluster, entities in grouped_entities.items():
        # Combine entities into a single message
        message = f'''The following is a list of entities in a cluster:\n{', '.join(entities)}\n\nPlease provide a descriptive name for this cluster.
        Just give only the name without any other text or symbols'''
        
        print(f"Processing Cluster {cluster}...")
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}]
        )
        cluster_name = chat.choices[0].message.content.strip()
        cluster_names[cluster] = cluster_name
        # break

    # Save cluster names to a CSV
    print(f"Saving cluster names to {cluster_names_file}...")
    df_cluster_names = pd.DataFrame(
        [{"Cluster": cluster, "Cluster Name": name, "Task": grouped_tasks[cluster]} for cluster, name in cluster_names.items()]
    )
    df_cluster_names.to_csv(cluster_names_file, index=False)
    print(f"Cluster names saved to {cluster_names_file}")

    return cluster_names

# --- Step 6: Link Tasks to Clusters ---
def link_tasks_to_clusters(tasks_output_file, cluster_names_file, updated_tasks_file):
    """
    Link tasks to clusters based on the tasks in the cluster_names_file and save the updated tasks with cluster names.

    Args:
        df_responses (pd.DataFrame): DataFrame with tasks and responses.
        cluster_names_file (str): Path to the file containing cluster names and tasks.
        output_file (str): Path to save the updated tasks with cluster names.
    """
    # Load cluster names and tasks
    df_cluster_names = pd.read_csv(cluster_names_file)
    df_responses = pd.read_csv(tasks_output_file)

    # Create a mapping of tasks to cluster names
    task_to_cluster_name = {}
    for _, row in df_cluster_names.iterrows():
        cluster_name = row["Cluster Name"]
        tasks_set = eval(row["Task"])  # Convert string representation of set back to actual set
        for task in tasks_set:
            task_to_cluster_name[task.strip()] = cluster_name

    # Add a "Cluster Name" column to df_responses
    cluster_names_per_task = []
    for _, row in df_responses.iterrows():
        task = row["Task"].strip() if pd.notna(row["Task"]) else None
        if task and task in task_to_cluster_name:
            cluster_names_per_task.append(task_to_cluster_name[task])
        else:
            cluster_names_per_task.append("")

    df_responses["Cluster Name"] = cluster_names_per_task

    # Save the updated DataFrame to a CSV
    df_responses.to_csv(updated_tasks_file, index=False)
    print(f"Updated task responses with cluster names saved to {updated_tasks_file}")



# --- Execution Pipeline ---
if __name__ == "__main__":
    # File paths
    input_file = 'D:/Seva-Allocation/Clustering/data/input_participant_info_cleaned.csv'
    tasks_output_file = 'D:/Seva-Allocation/Clustering/data/Task_Responses.csv'
    entities_output_file = 'D:/Seva-Allocation/Clustering/data/Entities.csv'
    nodes_output_file = 'D:/Seva-Allocation/Clustering/data/Embedded_Nodes.csv'
    clusters_output_file = 'D:/Seva-Allocation/Clustering/data/Clusters.csv'
    dendrogram_output_file = 'D:/Seva-Allocation/Clustering/Work-Experience-Tasks-dendrogram.png'
    cluster_names_file = 'D:/Seva-Allocation/Clustering/data/Cluster_Names.csv'
    updated_tasks_file = 'D:/Seva-Allocation/Clustering/data/Updated_Task_Responses.csv'

    # Step 1: Get Cleaned Tasks
    tasks_sample = get_cleaned_tasks(input_file, 'Work Experience/Tasks', sample_size=20)

    # Step 2: Fetch Task List from ChatGPT
    fetch_task_list(tasks_sample, tasks_output_file, entities_output_file)

    # Step 3: Create Nodes with Embeddings
    create_nodes_with_embeddings(tasks_output_file, nodes_output_file)

    # Step 4: Perform Clustering
    grouped_tasks, grouped_entities, total_clusters = perform_hierarchical_clustering(
        nodes_output_file,
        clusters_output_file,
        dendrogram_output_file,
        distance_threshold=0.21  # Increase if you want less clusters, decrease for more clusters
    )

    # Step 5: Name Clusters with GPT
    cluster_names = name_clusters_with_gpt(grouped_tasks, grouped_entities, client, cluster_names_file)

    # Step 6: Link Tasks to Clusters
    link_tasks_to_clusters(tasks_output_file, cluster_names_file, updated_tasks_file)
