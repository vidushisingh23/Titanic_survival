Titanic Survival Visualization-

About
This project visualizes survival patterns in the Titanic dataset (train_data.csv). It creates plots to explore factors like passenger class, gender, age, fare, and embarkation port. The project is stored in the titanic_survival folder.

Files in This Repository-
assign3.py: The Python script that generates the visualizations.

data/train_data.csv: The Titanic dataset used for analysis.

requirements.txt: Lists the Python libraries needed to run the script.

README.md: This file with instructions.

How to Run the Project

Clone the Repository:

git clone https://github.com/vidushisingh23/Titanic_survival
cd titanic-survivale

Install Required Libraries: Make sure you have Python 3.6 or higher installed. Then, install the libraries:

pip install -r requirements.txt

Check the Dataset: The train_data.csv file should be in the data/ folder. If it’s missing, download the Titanic dataset from Kaggle Titanic Dataset, rename it to train_data.csv, and place it in the data/ folder.

Run the Script: Run the script to create the visualizations:

python assign3.py

The script will generate plots and save them as PNG files in the output folder.

What the Script Does
The script creates the following visualizations:

Survival rates by passenger class (bar plot)
Survival distribution by gender (pie charts)
Age distribution by survival and gender (box plot)
Correlation between numerical features (heatmap)
Fare distribution by class and survival (violin plot)
Survival by embarkation port (stacked bar plot)

Notes-
Make sure the output folder exists before running the script.

The plots are saved as PNG files with a resolution of 300 DPI.

If you run into issues, the script will show error messages to help you troubleshoot.

Thank you for reviewing my project!