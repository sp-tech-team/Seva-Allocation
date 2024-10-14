

# Seva Allocation AI Assistant

The Seva Allocation AI Assistant is a powerful tool designed to streamline and optimize the process of Seva Allocation for the Sadhanpada Seva Team. Leveraging advanced data processing techniques and machine learning algorithms, this assistant aids in extracting skills from participants' data, ranking them to different job titles, and categorizing job titles into a structured hierarchy of clusters. This enhances the Seva Team's efficiency of allocating tasks, roles, and responsibilities (Seva) across different departments or sectors.

This command-line interface (CLI) application processes Excel files containing VRF data, SEVA data, LinkedIn data--and, optionally, educational mappings and job clusters--to generate predictions, extracted skills, and summary data.

# File Structure

Project_Root/
│
├── TOY_DATA
│   ├── vrf_data.xlsx
│   ├── seva_data.xlsx
│   └── linkedin_data.xlsx
│
├── 
│   └── SEVA_ALLOCATION_v1.py
│
└── SEVA_ALLOCATION_OUTPUT
    ├── predictions_data.xlsx
    ├── extracted_skills_data.xlsx
    └── summary_data.xlsx


# Use Cases

Task Allocation: Optimizes the distribution of tasks and responsibilities across various departments or sectors by accurately categorizing job titles.

Data Management: Enhances data management and organization by providing a structured approach to job title classification and cluster analysis.

## Requirements

- Python 3.6 or later
- Required Python libraries: `pandas`, `numpy`, `sentence_transformers`

## Installation

1. Ensure Python 3.6 or later is installed on your system.
2. Clone this repository or download the source code.
3. Navigate to the project directory and install the required Python libraries using:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

Navigate to the project directory and execute the script from the command line, specifying the paths to your input files:

```sh
python SEVA_ALLOCATION_v1.py <vrf_data_path> <seva_data_path> <linkedin_data_path> [--degree_to_skill_mapping <mapping_path>] [--job_clusters_latest <clusters_path>] [--predictions_data <predictions_path>] [--extracted_skills_data <skills_path>] [--summary_data <summary_path>]
```

### Arguments

- `<vrf_data_path>`: Path to the VRF data file (xlsx or csv).
- `<seva_data_path>`: Path to the SEVA data file (xlsx or csv).
- `<linkedin_data_path>`: Path to the LinkedIn data file (xlsx or csv).
- `--degree_to_skill_mapping <mapping_path>`: (Optional) Path to the degree to skill mapping file (json).
- `--job_clusters_latest <clusters_path>`: (Optional) Path to the job clusters file (txt).
- `--predictions_data <predictions_path>`: (Optional) Path to save predictions data (xlsx).
- `--extracted_skills_data <skills_path>`: (Optional) Path to save extracted skills data (xlsx).
- `--summary_data <summary_path>`: (Optional) Path to save summary data (xlsx).

### Examples

```sh
python SEVA_ALLOCATION_v1.py ./data/vrf_data.xlsx ./data/seva_data.xlsx ./data/linkedin_data.xlsx

WINDOWS WEBINAR MACHINE: 
FROM THIS DIRECTORY:
C:\Users\prantika.goswami\Downloads\seva_code_FINAL\seva_code_FINAL

RUN THIS CODE IN CMD PROMPT:
python SEVA_ALLOCATION_V1_FINAL.py TOY_DATA/TOY_vrf_data.xlsx TOY_DATA/TOY_seva_modified2.xlsx TOY_DATA/TOY_linkedin_data.xlsx

```

## Output

The script generates three Excel files:

- Predictions Data (`predictions_data.xlsx`)
- Extracted Skills Data (`extracted_skills_data.xlsx`)
- Summary Data (`summary_data.xlsx`)

By default, these files are saved in the current directory. Paths can be customized using optional arguments.



## Schemas

OG_SEVA_DF (INPUT DATAFRAME #1)
- **SP ID**: Integer
- **Registration Batch**: String
- **Gender**: String
- **Age**: Float
- **Seva Dept**: String
- **City**: String
- **State**: String
- **Nationality**: String
- **Country**: String
- **Work Experience/Company**: String
- **Work Experience/Designation**: String
- **Work Experience/Tasks**: String
- **Work Experience/Industry**: String
- **Work Experience/From Date**: String
- **Work Experience/To Date**: String
- **Education/Qualifications**: String
- **Education/Institution's Name**: String
- **Education/City**: String
- **Education/Specialization**: String
- **Education/Year of Passing/Graduation**: String
- **Marital status**: String
- **Program Tags**: String
- **Are you Coming with Laptop?**: String
- **Are you coming as couple?**: String
- **Add'l Skills**: Float
- **Add'l Skills.1**: Float
- **Any Additional Skills**: String
- **Computer Skills**: String
- **Skills**: String
- **Skills.1**: String
- **Program History**: String
- **Program History.1**: String
- **Local Volunteering**: String
- **Local Volunteering(days)**: Float
- **Are you currently taking any medication? If yes, list medication & its purpose**: String
- **Are you currently taking any medication? If yes, list medication & its purpose.1**: String
- **Have you taken any medication in the past? If yes, list medication and its purpose here**: String
- **Have you taken any medication in the past? If yes, list medication and its purpose here.1**: String
- **Highlights to SP Team/Value**: String
- **Interviewer Feedback**: String
- **Interviewer Feedback/Answer**: String
- **Input from the interviewer**: String
- **Concerns**: String
- **Please enter any concerns here**: String
- **Languages**: String
- **Languages/Can read**: String
- **Languages/Can speak**: String
- **Languages/Can type**: String
- **Languages/Can write**: String
- **Volunteering at IYC**: String
- **Volunteering at IYC/Volunteering Duration (No. of Days)**: String
- **Volunteering at IYC/Center Activity**: String
- **Volunteering at IYC/Description**: String
- **Local Volunteering/Volunteering Duration (No. of Days)**: String
- **Local Volunteering/Local center activity**: String
- **Any Hobbies/Interests**: String
- **Hobbies/Interests/Type**: String
- **Hobbies/Interests/Name**: String
- **Isha Connect/Name**: String

LINKEDIN_DF (INPUT DATAFRAME #2) 
- **SP ID**: Integer
- **Person Id**: Integer
- **Registration Batch**: String
- **Full Name**: String
- **Designation**: String
- **Featured Company/Institute**: String
- **Followers**: String
- **Connections**: Float
- **About**: String
- **Last Posted**: String
- **Top five endorsed skills**: String
- **All Skills**: String
- **Location**: String
- **Most Recently Studied Institute**: String
- **Most recently obtained Degree**: String
- **Most recently obtained Degree start year**: String
- **Most recently obtained Degree end year**: String
- **Most recent company**: String
- **Designation in Most recent company**: String
- **Most recent company location**: String
- **Most recent company start year**: Float
- **Most recent company end year**: Float
- **Is Currently Working**: String
- **First company**: String
- **Designation in First company**: String
- **First company location**: String
- **First company start year**: Float
- **First company end year**: Float
- **Overall total Experience(years)**: Integer
- **URL**: String

VRF_DF (INPUT DATAFRAME #3)
- **Department**: String
- **/**: String
- **Job Description**: String
- **Request Name**: String
- **Gender Preference**: String
- **# of Volunteers**: Float
- **Work Experience Needed?**: Float
- **Number of Years**: String
- **Skills/Keywords**: String
- **Educational Qualification**: String


