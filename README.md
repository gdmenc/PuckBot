# PuckBot
PuckBot: Predictive Interception and Striking Control for IIWA Air Hockey

# Requirements and Development
Create a new conda environment using conda create -n puckbot python=3.11 pip. Then activate this environment with conda activate puckbot.
Install the required packages using python -m pip install -r requirements.txt. 

#How to Run
Use the puckbot conda environment.

2-Arm Scenario (Default)
Run a 2-arm game for 30 seconds:

python run.py --num_arms 2 --game_duration 30.0
1-Arm Scenario
Run a single-robot scenario (Right arm only):

python run.py --num_arms 1 --game_duration 30.0