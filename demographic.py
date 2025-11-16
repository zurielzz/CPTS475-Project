import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# demographic analysis script
print("--- STARTING DEMOGRAPHIC ANALYSIS ---")

# clean data
try:
    df_clean = pd.read_csv('cleaned_survey_data.csv')
except FileNotFoundError:
    print("FATAL ERROR: 'cleaned_survey_data.csv' not found.")
    print("Please run your main 'analysis.py' script first.")
    sys.exit()

# --- sentiment chart by what college the student is in, engineering,art, or something else ---
print("Generating plot: Sentiment by College...")
plt.figure(figsize=(12, 8))
sns.countplot(
    data=df_clean,
    y='Q11',  # Use y-axis for long college names
    hue='sentiment_label',
    order=df_clean['Q11'].value_counts().index # Order by most common college
)
plt.title('Sentiment Breakdown by WSU College', fontsize=16)
plt.xlabel('Number of Students', fontsize=12)
plt.ylabel('College', fontsize=12)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('analysis_plot_college.png')
print("Saved 'analysis_plot_college.png'")

# --- sentiment by year, junior ,senior etc ---
print("Generating plot: Sentiment by Year...")
plt.figure(figsize=(10, 6))
sns.countplot(
    data=df_clean,
    x='Q10', # Use x-axis for 'Year'
    hue='sentiment_label',
    order=['Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate Student'] # Logical order
)
plt.title('Sentiment Breakdown by Year of Study', fontsize=16)
plt.xlabel('Year of Study', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.savefig('analysis_plot_year.png')
print("Saved 'analysis_plot_year.png'")

print("\n--- DEMOGRAPHIC ANALYSIS COMPLETE ---")