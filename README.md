# GNN_PN_Operation_MSc_Thesis

Matthijs de Jong's MSc Thesis project on topology control with graph neural networks (GNNs).

The project consists of a pipeline of four distinct parts:
1. Application of the rule-based agents that generates raw teacher data
2. Processing and data analysis of the teacher data
3. Training the ML models on the teacher data
4. Application of the ML models

~Unfortunately, loading modules from a parent directory is a nightmare in Python. Hence, to use some scripts in child directories (e.g `Tutor/Generate_teaching_dataset.py`) either require a IDE where you can easily run scripts from another wdir (such as Spyder) or moving the scripts to the root directory to execute them.~
