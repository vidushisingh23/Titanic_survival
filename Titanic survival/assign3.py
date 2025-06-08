import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
class TitanicVisualizer:
    def __init__(self, data_path="C:\\Users\\Sanjeev Kumar\\Desktop\\Titanic_survival\\data\\train_data.csv", output_dir="C:\\Users\\Sanjeev Kumar\\Desktop\\Titanic_survival\\outputs"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.survival_palette = ["#e74c3c", "#2ecc71"]  # Red: Perished, Green: Survived
        self.class_palette = ["#1abc9c", "#3498db", "#9b59b6"]  # Class colors
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        self.plot_counter = 0  

    def ensure_output_dir(self):
        print(f"ensure the output directory exists: {self.output_dir}")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} passenger records")
            return True
        except FileNotFoundError:
            print(f"Could not find {self.data_path}")
            return False
        except pd.errors.EmptyDataError:
            print("Empty dataset")
            return False

    def prepare_data(self):
        if self.df is None:
            return False
        # New features
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df['IsAlone'] = (self.df['FamilySize'] == 1).astype(int)
        self.df['AgeGroup'] = pd.cut(
            self.df['Age'],
            bins=[0, 12, 18, 30, 50, 100],
            labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Elderly']
        )
        self.df['Title'] = self.df['Name'].str.extract(' ([A-Za-z]+)', expand=False)
        self.df['FareBin'] = pd.qcut(self.df['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Handle missing values
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        
        print("Data preparation complete")
        return True

    def save_plot(self, filename_prefix):
        self.plot_counter += 1
        filename = f"{filename_prefix}_{self.plot_counter}.png"
        filepath = f"{self.output_dir}\\{filename}"
        plt.savefig(filepath, dpi=300)
        print(f"Saved visualization: {filename}")

    def plot_class_survival(self):
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Pclass', 
            y='Survived', 
            data=self.df, 
            palette=self.class_palette,
            errorbar=None
        )
        plt.title("Survival Rate by Passenger Class", fontsize=16, pad=20)
        plt.xlabel("Passenger Class", fontsize=12)
        plt.ylabel("Survival Rate", fontsize=12)
        plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])
        self.save_plot("class_survival")
        plt.close()

    def plot_gender_pie(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        colors = self.survival_palette
        # Male 
        male_data = self.df[self.df['Sex'] == 'male']['Survived'].value_counts()
        ax1.pie(male_data, labels=['Perished', 'Survived'], colors=colors, 
                autopct='%1.1f%%', startangle=90)
        ax1.set_title("Male Survival Distribution") 
        # Female
        female_data = self.df[self.df['Sex'] == 'female']['Survived'].value_counts()
        ax2.pie(female_data, labels=['Perished', 'Survived'], colors=colors, 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title("Female Survival Distribution")
        
        plt.suptitle("Survival Distribution by Gender", fontsize=16)
        self.save_plot("gender_survival_pie")
        plt.close()

    def plot_age_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x='Survived', 
            y='Age', 
            hue='Sex',
            data=self.df,
            palette='coolwarm'
        )
        plt.title("Age Distribution by Survival and Gender", fontsize=16)
        plt.xlabel("Survival Status", fontsize=12)
        plt.ylabel("Age", fontsize=12)
        plt.xticks([0, 1], ['Perished', 'Survived'])
        plt.axhline(18, color='gray', linestyle='--', alpha=0.5)
        plt.text(0.5, 16, '"Women and Children First"', 
                ha='center', fontsize=10, color='#2c3e50')
        self.save_plot("age_distribution")
        plt.close()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        numerical_cols = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Survived']
        correlation_matrix = self.df[numerical_cols].corr()
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            fmt='.2f'
        )
        plt.title("Correlation Heatmap of Numerical Features", fontsize=16)
        self.save_plot("correlation_heatmap")
        plt.close()

    def plot_fare_survival(self):
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            x='Pclass',
            y='Fare',
            hue='Survived',
            split=True,
            data=self.df,
            palette=self.survival_palette
        )
        plt.title("Fare Distribution by Class and Survival", fontsize=16)
        plt.xlabel("Passenger Class", fontsize=12)
        plt.ylabel("Fare", fontsize=12)
        plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])
        self.save_plot("fare_distribution")
        plt.close()

    def plot_embarkation_survival(self):
        plt.figure(figsize=(10, 6))
        embark_data = self.df.groupby(['Embarked', 'Survived']).size().unstack()
        embark_data.plot(kind='bar', stacked=True, color=self.survival_palette)
        plt.title("Survival by Embarkation Port", fontsize=16)
        plt.xlabel("Embarkation Port", fontsize=12)
        plt.ylabel("Passenger Count", fontsize=12)
        plt.xticks([0, 1, 2], ['Southampton', 'Cherbourg', 'Queenstown'], rotation=0)
        plt.legend(['Perished', 'Survived'], title='Outcome')
        self.save_plot("embarkation_survival")
        plt.close()

    def run_visualizations(self):
        self.ensure_output_dir()
        if not self.load_data():
            return
        if not self.prepare_data():
            return
        self.plot_class_survival()
        self.plot_gender_pie()
        self.plot_age_distribution()
        self.plot_correlation_heatmap()
        self.plot_fare_survival()
        self.plot_embarkation_survival()
        print("\nVisualization done")

if __name__ == "__main__":
    print("\nTITANIC DATA VISUALIZATION")
    visualizer = TitanicVisualizer()
    visualizer.run_visualizations()