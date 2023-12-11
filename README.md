# GNN_PN_Operation_MSc_Thesis

Matthijs de Jong's MSc Thesis project on topology control with graph neural networks (GNNs).

The project consists of a pipeline of four distinct parts:
1. Application of the rule-based agents that generates raw teacher data
2. Processing and data analysis of the teacher data
3. Training the ML models on the processed teacher data
4. Evaluation of the trained ML models

## 0. Auxiliary functionality


## 1. Application of the rule-based agents

The code for applying the rule-based agents is mostly contained in three files:
- [`generate_imitation_data.py`](generate_imitation_data.py) is the script that starts the application process. It takes the following arguments:
  - `--do_nothing_capacity_threshold`: The threshold that, if not exceeded by the max. rho percentage, causes the agent to choose do-nothing actions.
  - `--disable_line`: The index of the line to be disabled. A value of '-1' disables no lines.
  - `--start_chronic_id`: The chronic/scenario number from which to start.
  - `--strategy`: Which rule-based agent to employ. Acceptable values are 'Greedy' or 'CheckNMinOne'.
- [`imitation_generation/generation.py`](imitation_generation/generation.py) contains the code for applying the agent and for saving the generated data. 
- [`imitation_generation/tutor.py`](imitation_generation/tutor.py) contains the functionality for the different rule-based agents. 

The generated data is also used to evaluate the performance of the rule-based agents.

## 2. Data processing and analysis


## 3. ML model training


## 4. ML model evaluation
TBD

~Unfortunately, loading modules from a parent directory is a nightmare in Python. Hence, to use some scripts in child directories (e.g `Tutor/Generate_teaching_dataset.py`) either require a IDE where you can easily run scripts from another wdir (such as Spyder) or moving the scripts to the root directory to execute them.~
