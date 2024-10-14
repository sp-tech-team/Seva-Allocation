#!/usr/bin/env python
# coding: utf-8
# Author: Roshan Ram

import argparse
import json

# Define the main function to process the data
def main(vrf_data_path, seva_data_path, linkedin_data_path, degree_to_skill_mapping_path, job_clusters_path, predictions_data_path, extracted_skills_data_path, summary_data_path):
    
    from tqdm import tqdm
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    import pandas as pd
    import re
    import ast
    
    # Load input data
    vrf_data_df = pd.read_excel(vrf_data_path)
    og_seva_df = pd.read_excel(seva_data_path)
    linkedin_df = pd.read_excel(linkedin_data_path)

    # Optional degree to skill mapping
    degree_profession_mapping = {}
    if degree_to_skill_mapping_path:
        with open(degree_to_skill_mapping_path, "r") as json_file: # "degree_profession_mapping.json"
            degree_profession_mapping = json.load(json_file)
    
    
    # Function to read and convert JOB_CLUSTERS_LATEST from a text file
    def read_job_clusters_latest(file_path):
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    # Reading and processing the JOB_CLUSTERS_LATEST file
    JOB_CLUSTERS_LATEST = read_job_clusters_latest(args.job_clusters_latest)

    # Combine and process data
    master_sheet_df = og_seva_df.copy()
    master_sheet_df['SKILLS_ALL'] = og_seva_df['Any Additional Skills'].fillna(' ') + ' ' + og_seva_df['Computer Skills'].fillna(' ') + ' ' + og_seva_df['Skills'].fillna(' ') + ' ' + og_seva_df['Skills.1'].fillna(' ')
    master_sheet_df['WORK_EXPERIENCE_ALL'] = np.NaN
    for work_col in ['Work Experience/Company', 'Work Experience/Designation', 'Work Experience/Tasks', 'Work Experience/Industry', 'Work Experience/From Date', 'Work Experience/To Date']:
        master_sheet_df['WORK_EXPERIENCE_ALL'] = master_sheet_df['WORK_EXPERIENCE_ALL'].fillna(' ') + ' ' + master_sheet_df[work_col].fillna(' ')
    master_sheet_df['WORK_EXPERIENCE_ALL'] = master_sheet_df['WORK_EXPERIENCE_ALL'].apply(lambda s: s.strip())


    master_sheet_df['HOBBIES_ALL'] = np.NaN
    for hobbies_col in ['Any Hobbies/Interests', 'Hobbies/Interests/Type', 'Hobbies/Interests/Name']:
        master_sheet_df['HOBBIES_ALL'] = master_sheet_df['HOBBIES_ALL'].fillna(' ') + ' ' + master_sheet_df[hobbies_col].fillna(' ')



    master_sheet_df['CONCERNS_ALL'] = master_sheet_df['Concerns'].fillna(' ') + ' ' + master_sheet_df['Please enter any concerns here'].fillna(' ')


    master_sheet_df['LANGUAGES_ALL'] = np.NaN

    for lang_col in ['Languages',
     'Languages/Can read',
     'Languages/Can speak',
     'Languages/Can type',
     'Languages/Can write']:
        master_sheet_df['LANGUAGES_ALL'] = master_sheet_df['LANGUAGES_ALL'].fillna(' ') + ' ' + master_sheet_df[lang_col].fillna(' ')

    master_sheet_df['LANGUAGES_ALL'] = master_sheet_df['LANGUAGES_ALL'].apply(lambda s : s.strip())


    # In[158]:


    master_sheet_df.rename(columns={"Concerns":"CONCERNS_ALL", "Gender":"GENDER_ALL", "Age":"AGE_ALL"}, inplace=True)


    # In[160]:


    linkedin_df.rename(columns={'SP ID':'SP_ID'}, inplace=True)

    linkedin_df = linkedin_df.add_suffix('_LINKEDIN')
    master_sheet_df = master_sheet_df.add_suffix('_APPLICATIONFORM')

    linkedin_df.rename(columns={'SP_ID_LINKEDIN':'SP_ID'}, inplace=True)
    master_sheet_df.rename(columns={'SP ID_APPLICATIONFORM':'SP_ID'}, inplace=True)


    # In[6]:


    # new_df = pd.read_excel('../data/skill_comparison_output_ALL_ROWS_phase2_v1.xlsx')


    # In[165]:


    import pandas as pd

    # List of unwanted skills
    unwanted_skills = [
        'Basic Computer Skills / Basic Computer (MS Office and Email) Skills',
        'Soft Skills / Fit for Physical Seva',
        'Soft Skills / Soft-spoken and cordial',
        'Soft Skills / Enthusiastic',
        'Soft Skills / Articulate in communication'
    ]

    # Split skills on '\n' and reset index to preserve 'SP_ID'
    skills_df = og_seva_df[['SP ID', 'Skills']].assign(Skills=og_seva_df['Skills'].str.split('\n')).explode('Skills').reset_index(drop=True)

    # Filter unwanted skills
    filtered_skills = skills_df[~skills_df['Skills'].isin(unwanted_skills)].reset_index(drop=True)

    # Concatenate remaining skills with comma
    cleaned_skills = filtered_skills.groupby('SP ID')['Skills'].agg(', '.join).reset_index(name='Cleaned_Skills')

    # Merge with master_sheet_df on 'SP_ID'
    result_df = master_sheet_df.merge(cleaned_skills, left_on='SP_ID', right_on='SP ID', how='left')


    # In[166]:


    result_df.drop(columns=['SP ID'], inplace=True)


    # In[167]:


    master_sheet_df = result_df.rename(columns={'Cleaned_Skills':'INTERVIEWER_SKILLS'})


    # In[169]:


    # import pandas as pd

    # # Filter og_seva_df based on the condition of skill value counts
    # filtered_og_seva_df = og_seva_df[og_seva_df['Skills'].map(og_seva_df['Skills'].value_counts()) > 189]

    # # Join master_sheet_df with the filtered og_seva_df on "SP_ID"
    # # master_sheet_df.merge(filtered_og_seva_df[['SP ID', 'Skills']], left_on='SP_ID', right_on='SP ID', how='inner')

    # import pandas as pd

    # # Splitting the 'Skills' column in og_seva_df
    # skills_split = og_seva_df['Skills'].str.split('\n', expand=True).stack()

    # # Remove specific skills
    # skills_split = skills_split[~skills_split.isin(['Basic Computer Skills', 'Basic Computer (MS Office and Email) Skills',
    #                                                  'Soft Skills', 'Fit for Physical Seva',
    #                                                  'Soft-spoken and cordial', 'Enthusiastic',
    #                                                  'Articulate in communication'])]

    # # Join the remaining skills with a comma
    # result_skills = skills_split.groupby(level=0).agg(','.join).reset_index(name='Skills')

    # # Merge the result back to the master_sheet_df based on the 'SP_ID' key
    # result_df = master_sheet_df.merge(result_skills, left_on='SP_ID', right_index=True, how='left')

    # # Fill NaN values in the 'Skills' column with an empty string
    # # result_df['Skills'] = result_df['Skills'].fillna('')

    # # Display the result
    # print(result_df)


    # Create a mapping table to map education to a skill. Need to map each degree to a skill.  
    #         - E.g. M Ayurvedic Medicines = Ayurvedic Doctor

    # In[170]:


    # In[173]:


    # degree_profession_mapping = {**degree_profession_mapping1, **degree_profession_mapping2}


    # In[174]:


    # Function to map degrees to professions
    degrees_not_mapped = [ ]
    def map_degrees_to_professions(degrees):
        skills_list = []
        for degree in degrees:
            if degree in degree_profession_mapping:
                skills_list.extend(degree_profession_mapping[degree])
            else:
                degrees_not_mapped.append('Degree Not Found: ' + degree)
        return ', '.join(skills_list)


    # In[175]:


    # Apply the mapping function to create the 'EDUCATION_SKILLS' column
    og_seva_df['EDUCATION_SKILLS'] = og_seva_df['Education/Qualifications'].apply(
        lambda x: map_degrees_to_professions(str(x).split('\n')) if not pd.isna(x) else ''
    )


    # In[176]:


    merged_df = pd.merge(master_sheet_df, og_seva_df[['SP ID', 'EDUCATION_SKILLS']], left_on='SP_ID', right_on='SP ID', how='inner')


    # In[177]:


    master_sheet_df = merged_df.drop(columns='SP ID')


    # ### Add DESIGNATION_SKILLS

    # In[178]:


    # Step 0: Identify and handle duplicates and NaN values in 'Work Experience/Designation' column
    og_seva_df['Work Experience/Designation'] = og_seva_df['Work Experience/Designation'].apply(lambda x: x.split('\n') if isinstance(x, str) else x)
    og_seva_df_exploded = og_seva_df.explode('Work Experience/Designation')
    og_seva_df_exploded.dropna(subset=['Work Experience/Designation'], inplace=True)
    og_seva_df_exploded.drop_duplicates(subset=['SP ID', 'Work Experience/Designation'], inplace=True)

    # Step 1: Group by 'SP_ID' and join the designations using commas
    og_seva_df_grouped = og_seva_df_exploded.groupby('SP ID')['Work Experience/Designation'].agg(lambda x: ', '.join(str(v) for v in x)).reset_index()

    # Check for duplicate 'SP_ID' values in og_seva_df_grouped
    duplicates_og_seva = og_seva_df_grouped[og_seva_df_grouped.duplicated('SP ID')]
    if not duplicates_og_seva.empty:
        raise ValueError(f'Duplicate SP_ID values found in og_seva_df_grouped: {duplicates_og_seva}')

    # Step 2: Reset index in 'master_sheet_df' if needed
    master_sheet_df_reset = master_sheet_df.reset_index()

    # Check for duplicate 'SP_ID' values in master_sheet_df_reset
    duplicates_master_sheet = master_sheet_df_reset[master_sheet_df_reset.duplicated('SP_ID')]
    if not duplicates_master_sheet.empty:
        raise ValueError(f'Duplicate SP ID values found in master_sheet_df_reset: {duplicates_master_sheet}')

    # Step 3: Merge with 'master_sheet_df_reset' on 'SP_ID'
    merged_df = pd.merge(master_sheet_df_reset, og_seva_df_grouped, left_on='SP_ID', right_on='SP ID', how='left')

    # Now 'merged_df' contains the desired result with designations separated by commas


    # In[179]:


    merged_df.rename(columns={'Work Experience/Designation':'DESIGNATION_SKILLS'}, inplace=True)


    # In[180]:


    merged_df.drop('SP ID', axis=1, inplace=True)


    # In[181]:


    master_sheet_df = merged_df


    # # RE-RUNNING SKILL COMPARISON

    # In[182]:


    df_vrf = vrf_data_df # pd.read_excel("./TOY_DATA/TOY_vrf_data.xlsx")#, encoding=result['encoding'])

    df_vrf = df_vrf[['Department', '/']].rename(columns = {'/':'Job Title'})


    # In[183]:


    import pandas as pd
    import ast  # For literal_eval function

    # Assuming 'GPT_SKILLS_ALL_APPLICATIONFORM', 'INTERVIEWER_SKILLS', and 'EDUCATION_SKILLS' are column names
    # def extract_skills(row):
    #     skills = []
    #     try:
    #         entries = ast.literal_eval(row)  # Safely evaluate the string as a list of dictionaries
    #         for entry in entries:
    #             if 'name' in entry:
    #                 skill = entry['name']
    #                 print(skill)
    #                 if skill != None: skills.append(skill)
    #     except (ValueError, SyntaxError):
    #         pass
    #     return ', '.join(skills) if (skills != None and len(skills)) else ''

    # def extract_skills(row):
    #     skills = []
    #     try:
    #         entries = ast.literal_eval(row)  # Safely evaluate the string as a list of dictionaries
    #         for entry in entries:
    #             if 'name' in entry:
    #                 skill = entry['name']
    # #                 print(skill)
    #                 if skill != None: skills.append(skill)
    #     except (ValueError, SyntaxError):
    #         pass
    #     if skills != None and len(skills):
    #         filtered_skills = [s for s in skills if s != '']
    #         joined_skills = ', '.join(filtered_skills)
    #         return joined_skills
    #     else:
    #         return ''


    # master_sheet_df['GPT_SKILLS'] = master_sheet_df['GPT_SKILLS_ALL_APPLICATIONFORM'].apply(extract_skills)
    # master_sheet_df['COMBINED_SKILLS'] = master_sheet_df.apply(
    #     lambda row: ','.join(
    #         filter(None, [
    #             ','.join(row['GPT_SKILLS']) if row['GPT_SKILLS'] else '',
    #             str(row['INTERVIEWER_SKILLS']) if pd.notna(row['INTERVIEWER_SKILLS']) else '',
    #             str(row['EDUCATION_SKILLS']) if pd.notna(row['EDUCATION_SKILLS']) else ''
    #         ])
    #     ),
    #     axis=1
    # )

    # # Convert the combined string to a Python list
    # master_sheet_df['COMBINED_SKILLS'] = master_sheet_df['COMBINED_SKILLS'].apply(lambda x: x.split(',') if x else [])

    # Drop the intermediate GPT_SKILLS column if needed
    # master_sheet_df = master_sheet_df.drop(columns=['GPT_SKILLS'])

    # Print or use the resulting DataFrame
    # print(master_sheet_df)


    # In[184]:


    # def concatenate_skills(row):
    #     combined_skills = (
    #         str(row['GPT_SKILLS']).strip(', ') +
    #         (', ' if not pd.isna(row['GPT_SKILLS']) and row['GPT_SKILLS'] != 'nan' and row['GPT_SKILLS'] != '' else '') +
    #         str(row['INTERVIEWER_SKILLS']).strip(', ') +
    #         (', ' if not pd.isna(row['INTERVIEWER_SKILLS']) and row['INTERVIEWER_SKILLS'] != 'nan' and row['INTERVIEWER_SKILLS'] != '' else '') +
    #         str(row['EDUCATION_SKILLS']).strip(', ') +
    #         (', ' if not pd.isna(row['EDUCATION_SKILLS']) and row['EDUCATION_SKILLS'] != 'nan' and row['EDUCATION_SKILLS'] != '' else '') +
    #         str(row['DESIGNATION_SKILLS']).strip(', ')

    #     )
    #     # Replace consecutive commas with a single comma
    #     return combined_skills.replace(', ', ', ')

    def concatenate_skills(row):
        skills_list = []

        # GPT_SKILLS
        # if not pd.isna(row['GPT_SKILLS']) and str(row['GPT_SKILLS']).strip() != '':
        #     skills_list.append(str(row['GPT_SKILLS']).strip(', '))

        # INTERVIEWER_SKILLS
        if not pd.isna(row['INTERVIEWER_SKILLS']) and str(row['INTERVIEWER_SKILLS']).strip() != '':
            skills_list.append(str(row['INTERVIEWER_SKILLS']).strip(', '))

        # EDUCATION_SKILLS
        if not pd.isna(row['EDUCATION_SKILLS']) and str(row['EDUCATION_SKILLS']).strip() != '':
            skills_list.append(str(row['EDUCATION_SKILLS']).strip(', '))

        # DESIGNATION_SKILLS
        if not pd.isna(row['DESIGNATION_SKILLS']) and str(row['DESIGNATION_SKILLS']).strip() != '':
            skills_list.append(str(row['DESIGNATION_SKILLS']).strip(', '))

        # Combine non-empty skills using ', '
        combined_skills = ', '.join(skills_list)

        return combined_skills


    # Apply the function to each row
    master_sheet_df['COMBINED_SKILLS'] = master_sheet_df.apply(concatenate_skills, axis=1)


    # In[185]:


    # Preprocess the data
    VRF_job_titles = df_vrf['Job Title'].tolist()

    # # Replace NaN values with empty strings
    # updated_VRF_job_titles = [ ]
    # for title in VRF_job_titles:
    #     if not pd.isna(title):
    #         updated_VRF_job_titles.append(title)
    # # VRF_job_titles = ["" if pd.isna(title) else title for title in VRF_job_titles]
    # VRF_job_titles = updated_VRF_job_titles

    # Preprocess the data
    df_vrf_cleaned = df_vrf.dropna(subset=['Job Title'])
    VRF_job_titles = df_vrf_cleaned['Job Title'].tolist()

    # Replace NaN values with empty strings
    updated_VRF_job_titles = ["" if pd.isna(title) else title for title in VRF_job_titles]
    VRF_job_titles = updated_VRF_job_titles


    # In[187]:


    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    
    print('EMBEDDING SKILLS...')

    job_title_corr_mat = np.zeros((len(master_sheet_df), len(VRF_job_titles)))
    embeddings_master = model.encode(master_sheet_df['COMBINED_SKILLS'], convert_to_tensor=True, show_progress_bar=True)


    # In[188]:
    
    print('EMBEDDING JOB TITLES...')


    embeddings_vrf = model.encode(VRF_job_titles, convert_to_tensor=True, show_progress_bar=True)
    
    print('COMPUTING COSINE SIMILARITY SCORES...')

    # # Compute cosine similarity scores
    cosine_scores = util.cos_sim(embeddings_master, embeddings_vrf)


    # In[189]:


    # Create a list to store the results
    
    print('SCORING AND MATCHING SKILLS TO JOB TITLES...')
    
    results = []

    # Iterate through SP_IDs and df_vrf rows to calculate skill scores
    for i, sp_id in enumerate(master_sheet_df['SP_ID']):
        for (j, row) in df_vrf.dropna(subset=['Job Title']).reset_index(drop=True).iterrows():
            department = row['Department']
            job_title = row['Job Title']

            # Ensure that the indices are within bounds
            if i < len(cosine_scores) and j < len(cosine_scores[i]):
                # Extract skills for the SP_ID (as per your previous code)
                participant_skills = master_sheet_df[master_sheet_df['SP_ID'] == sp_id]['COMBINED_SKILLS'].iloc[0]

                # Calculate the skill score
                raw_skill_score = cosine_scores[i, j]
                skill_score = 0 if pd.isna(raw_skill_score) else raw_skill_score

                # Store the results including SP_ID, row number, department, job title, and skill score
                results.append({
                    'SP_ID': sp_id,
                    'Row Number': j,
                    'Department': department,
                    'Job Title': job_title,
                    'Skill Score': skill_score
                })
            else:
                print(f"Warning: Index out of bounds - i={i}, j={j}")

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)


    # In[190]:


    results_df['Skill Score'] = results_df['Skill Score'].apply(lambda x: x.numpy())

    # results_df.to_csv('../data/SKILL_SCORES_V2_PHASE2_NOGPT.csv') 


    # In[191]:


    results_df_department_not_null = results_df[~results_df['Department'].isna()]

    results_df_department_not_null = results_df_department_not_null.groupby(['SP_ID', 'Department', 'Job Title', 'Skill Score']).size().reset_index(name="Count")

    results_df_department_not_null.drop(columns=['Count'], inplace=True)
    # results_df_department_not_null.head()


    # In[192]:


    # Grouping by 'SP_ID' and sorting each group by 'Skill Score' in descending order
    results_df_department_not_null = results_df_department_not_null.groupby('SP_ID', group_keys=False).apply(lambda group: group.sort_values('Skill Score', ascending=False))

    # Resetting the index after sorting
    results_df_department_not_null.reset_index(drop=True, inplace=True)


    # In[193]:


    # WRITE OUTPUT FILES
    # master_sheet_df.to_csv('../data/PHASE2_V2_MERGED_MASTER_SHEET.csv')
    # results_df_department_not_null.to_csv('../data/VIEWS/results_df_department_not_null_PHASE2_v2.csv')

    # In[195]:


    def invert_job_clusters(job_clusters):
        optimized_dict = {}
        for main_category, subcategories in job_clusters.items():
            for subcategory, titles in subcategories.items():
                for title in titles:
                    optimized_dict[title] = (main_category, subcategory)
        return optimized_dict


    # In[196]:


    def get_category_optimized(job_title, optimized_dict):
        return optimized_dict.get(job_title, ("Not Found", "Not Found"))


    # In[197]:


    # JOB_CLUSTERS_LATEST = None

    # In[198]:


    def pretty_tree_to_regular(pretty_str):
        tree = {}
        lines = pretty_str.strip().split('\n')

        # Regular expressions to identify levels based on indentation
        pattern = re.compile(r'^( *)(- )?(.*)')

        # Helper function to insert an item into the tree based on indentation levels
        def insert_into_tree(tree, levels, item):
            for level in levels[:-1]:
                tree = tree.setdefault(level, {})
            tree.setdefault(levels[-1], []).append(item)

        # Stack to keep track of the current hierarchy based on indentation
        hierarchy_stack = []
        for line in lines:
            indent, _, item = pattern.match(line).groups()
            level = len(indent) // 2  # Assuming two spaces per indent level

            # Adjust the current item by removing colon if we're at the first or second level
            if level <= 1 and item.endswith(':'):
                item = item[:-1]

            # If we're at a deeper level, just append to the stack
            if level > len(hierarchy_stack):
                hierarchy_stack.append(item)
            else:
                # If we're at a shallower level, reset the stack to that level
                hierarchy_stack = hierarchy_stack[:level]
                hierarchy_stack.append(item)

            # If it's a job title (prefixed with '- '), insert it into the tree
            if _:
                insert_into_tree(tree, hierarchy_stack[:-1], hierarchy_stack[-1])
                hierarchy_stack.pop()  # Remove the job title from the hierarchy stack

        return tree

    # # Example usage:
    # pretty_str = """
    # Creative and Media:
    #   Music and Audio:
    #     - Music Producer / Arranger
    #     - Mixing Engineer
    #   Content Creation and Writing:
    #     - Content Writer
    # Information Technology and Software:
    #   Development and Programming:
    #     - Software Developer
    # """

    TREE_FORMAT_JOB_CLUSTERS_LATEST = pretty_tree_to_regular(JOB_CLUSTERS_LATEST)
    OG_FORMAT_JOB_CLUSTERS_LATEST = invert_job_clusters(TREE_FORMAT_JOB_CLUSTERS_LATEST)


    # In[199]:


    results_df_department_not_null['Predicted Cluster'] = results_df_department_not_null['Job Title'].apply(lambda x: ' - '.join(OG_FORMAT_JOB_CLUSTERS_LATEST.get(x, ("Not Found", "Not Found"))))


    # # OUTPUT TABLE #1 - PREDICTIONS TABLE

    # In[200]:


    results_df_department_not_null


    # # OUTPUT TABLE #2 - EXTRACTED SKILLS TABLE

    # In[201]:


    SKILLS_COLS = ['SP_ID', 'INTERVIEWER_SKILLS', 'EDUCATION_SKILLS', 'COMBINED_SKILLS']
    master_sheet_df[SKILLS_COLS]


    # # OUTPUT TABLE #3

    # In[202]:


    SUMMARY_COLS = ['SP ID',
     'Education/Qualifications','Education/Institution\'s Name', 'Education/City', 'Education/Specialization', 'Education/Year of Passing/Graduation', 
     'Work Experience/Designation', 'Work Experience/Tasks', 'Work Experience/Industry', 'Work Experience/From Date', 'Work Experience/To Date',
     'Interviewer Feedback/Answer']

    SUMMARY_TABLE = og_seva_df[SUMMARY_COLS]

    
    
    # Save output data
    # (Save your processed data frames to files as specified in the function arguments)
    # Example:
    print('SAVING OUTPUT...')
    results_df_department_not_null.to_excel(predictions_data_path)
    master_sheet_df[SKILLS_COLS].to_excel(extracted_skills_data_path)
    SUMMARY_TABLE.to_excel(summary_data_path)
    # Save other outputs as needed

# Setup argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input files and generate predictions, extracted skills, and summary data.")
    parser.add_argument("vrf_data", help="Path to VRF data file (xlsx or csv)")
    parser.add_argument("seva_data", help="Path to SEVA data file (xlsx or csv)")
    parser.add_argument("linkedin_data", help="Path to LINKEDIN data file (xlsx or csv)")
    parser.add_argument("--degree_to_skill_mapping", help="Path to DEGREE TO SKILL MAPPING file (json)", default=r'./TOY_DATA/degree_profession_mapping.json')
    parser.add_argument("--job_clusters_latest", help="Path to JOB_CLUSTERS_LATEST file (txt)", default=r'./TOY_DATA/job_clusters_latest.txt')

    parser.add_argument("--predictions_data", help="Path to save PREDICTIONS_DATA (xlsx)", default=r"./SEVA_ALLOCATION_OUTPUT/predictions_data.xlsx")
    parser.add_argument("--extracted_skills_data", help="Path to save EXTRACTED_SKILLS_DATA (xlsx)", default=r"./SEVA_ALLOCATION_OUTPUT/extracted_skills_data.xlsx")
    parser.add_argument("--summary_data", help="Path to save SUMMARY_DATA (xlsx)", default="./SEVA_ALLOCATION_OUTPUT/summary_data.xlsx")

    args = parser.parse_args()

    main(args.vrf_data, args.seva_data, args.linkedin_data, args.degree_to_skill_mapping, args.job_clusters_latest, args.predictions_data, args.extracted_skills_data, args.summary_data)