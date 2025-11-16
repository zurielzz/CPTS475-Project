import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import re
import numpy as np
import sys
from wordcloud import WordCloud
import plotly.graph_objects as go




nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
print("NLTK data download complete.")

# --- Load Data ---
file_name = 'survey_data.csv'
try:
    df = pd.read_csv(file_name, header=0, skiprows=[1, 2])
except FileNotFoundError:
    print(f"FATAL ERROR: '{file_name}' not found.")
    print("Please make sure your CSV file is in the same folder and named correctly.")
    sys.exit()

# --- Filter Data ---
df = df[df['Status'] == 'IP Address'].reset_index(drop=True)
df = df[df['Finished'] == True]
print(f"Successfully loaded and filtered data. Found {len(df)} real responses.")
if len(df) == 0:
    print("FATAL ERROR: No real responses found. Check your filters.")
    sys.exit()

# --- Run Sentiment Analysis ---
q9_column = 'Q9'
df[q9_column] = df[q9_column].fillna('')
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']
def get_sentiment_label(score):
    if score >= 0.05: return 'Positive'
    elif score <= -0.05: return 'Negative'
    else: return 'Neutral'

print(f"Running sentiment analysis on column: {q9_column}...")
df['sentiment_score'] = df[q9_column].apply(get_sentiment)
df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
df.to_csv('cleaned_survey_data.csv', index=False)
print("Saved 'cleaned_survey_data.csv'.")

df_clean = pd.read_csv('cleaned_survey_data.csv')

# ---  (Descriptive Stats) ---
print("\nLevel 1: Generating descriptive plots...")
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment_label', data=df_clean, order=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Breakdown of WSU Students')
plt.ylabel('Number of Responses')
plt.xlabel('Sentiment')
plt.savefig('level_1_sentiment_plot.png')
print("Saved 'level_1_sentiment_plot.png'")

# --- (Feature Importance) ---
print("\nLevel 2: Building Random Forest with Cross-Validation...")
feature_columns = [
    'Q3', 'Q4', 'Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5',
    'Q6a', 'Q6b', 'Q7_1', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5', 'Q7_6', 'Q7_7',
    'Q8', 'Q10', 'Q11'
]
df_model = df_clean[feature_columns].fillna('Missing')
for col in ['Q5_1', 'Q5_2', 'Q5_3', 'Q5_4', 'Q5_5', 'Q7_1', 'Q7_2', 'Q7_3', 'Q7_4', 'Q7_5', 'Q7_6', 'Q7_7']:
    df_model[col] = df_clean[col].fillna(0)
X = pd.get_dummies(df_model, drop_first=True)
y = df_clean['sentiment_label']

if len(X) >= 20: 
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf_model, X, y, cv=10)
    print("\n--- CROSS-VALIDATION ACCURACY ---")
    print(f"MEAN ACCURACY: {np.mean(cv_scores)*100:.2f}%")
    rf_model.fit(X, y) 
    importances = rf_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    label_map = {
        'Q6b_Missing': 'Is a Non-User (Personal)', 'Q6a_Missing': 'Is a Non-User (Academic)',
        'Q7_3_Finding articles, links, or resources': 'Task: Finding Resources',
        'Q6a_About once a week': 'Academic Use: Once a week',
        'Q7_5_Understanding a complex topic or concept': 'Task: Understanding Concepts',
        'Q5_5_Other': 'Tool: Other', 'Q4_Missing': 'Frequency: Missing',
        'Q7_4_Brainstorming ideas': 'Task: Brainstorming',
        'Q11_Other / I don\'t know': 'College: Other',
        'Q7_2_Writing essays, reports, or other text': 'Task: Writing Essays',
        'Q5_2_Gemini (Google)': 'Tool: Gemini', 'Q7_6_Writing emails': 'Task: Writing Emails',
        'Q6b_Multiple times a week': 'Personal Use: Multiple times/week',
        'Q8_Significantly increased it': 'Productivity: Significantly Increased',
        'Q7_7_Video/Image generation': 'Task: Image Generation'
    }
    top_features_df = importance_df.sort_values(by='Importance', ascending=False).head(15)
    top_features_df['Clean_Label'] = top_features_df['Feature'].apply(lambda x: label_map.get(x, x))
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Clean_Label', data=top_features_df) 
    plt.title('Top 15 Most Important Factors Predicting Sentiment')
    plt.ylabel('Feature') 
    plt.tight_layout()
    plt.savefig('level_2_feature_importance_CLEAN.png') 
    print("\nSaved 'level_2_feature_importance_CLEAN.png'")
else:
    print("Warning: Not enough data for reliable cross-validation. Skipping model.")


# word clouds 
print("\nLevel 3: Generating Word Clouds...")
stop_words = set(stopwords.words('english'))
positive_text = ' '.join(df_clean[df_clean['sentiment_label'] == 'Positive'][q9_column])
negative_text = ' '.join(df_clean[df_clean['sentiment_label'] == 'Negative'][q9_column])

if len(positive_text) > 0:
    pos_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens', stopwords=stop_words).generate(positive_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(pos_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top Positive Sentiment Words')
    plt.savefig('level_3_positive_wordcloud.png')
    print("Saved 'level_3_positive_wordcloud.png'")

if len(negative_text) > 0:
    neg_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds', stopwords=stop_words).generate(negative_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(neg_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Top Negative Sentiment Words')
    plt.savefig('level_3_negative_wordcloud.png')
    print("Saved 'level_3_negative_wordcloud.png'")

# ---(Clustering) ---
print("\nLevel 4: Finding user clusters...")
if len(df_clean) > 3:
    usage_features = ['Q4', 'Q6a', 'Q6b', 'Q8']
    df_cluster = df_clean[usage_features].fillna('Missing')
    X_cluster = pd.get_dummies(df_cluster, drop_first=True)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['cluster'] = kmeans.fit_predict(X_cluster)
    cluster_modes = df_clean.groupby('cluster')['Q4'].apply(lambda x: x.mode(dropna=False).iloc[0]).to_dict()
    non_user_cluster_id = -1
    for k, v in cluster_modes.items():
        if pd.isna(v): non_user_cluster_id = k
            
    def name_clusters(row):
        if row['cluster'] == non_user_cluster_id:
            return 'Usage Profile: Non-User'
        elif row['Q4'] == 'Daily':
            return 'Usage Profile: High User (Daily)'
        else:
            return 'Usage Profile: Moderate User (Weekly)'
            
    df_clean['User_Group'] = df_clean.apply(name_clusters, axis=1)
    print("User clusters identified.")
else:
    print("Not enough data to perform clustering.")

# --- Sankey Diagram ---
print("\nLevel 5: Generating Upgraded Sankey Diagram...")

df_clean['Q11'] = df_clean['Q11'].fillna('College Not Stated')

# 1. Manually define the order of our labels
all_college_names = list(df_clean['Q11'].unique())
main_colleges = [c for c in all_college_names if c not in ["Other / I don't know", "College Not Stated"]]
main_colleges.sort()
colleges = main_colleges + ["Other / I don't know", "College Not Stated"]
user_groups = ['Usage Profile: High User (Daily)', 'Usage Profile: Moderate User (Weekly)', 'Usage Profile: Non-User']
sentiments = ['Positive', 'Neutral', 'Negative']
all_labels = list(colleges) + list(user_groups) + list(sentiments)

# 2. Create a color map for all labels
college_colors = ['#EF476F'] * len(colleges) # Pink
group_colors = ['#FFD166'] * len(user_groups) # Yellow
sentiment_map = {'Positive': '#06D6A0', 'Neutral': '#118AB2', 'Negative': '#EF476F'} # Green, Blue, Red
sentiment_colors = [sentiment_map[s] for s in sentiments]
all_colors = college_colors + group_colors + sentiment_colors

# 3. Build the the flows
links_c_ug = df_clean.groupby(['Q11', 'User_Group']).size().reset_index(name='count')
links_ug_s = df_clean.groupby(['User_Group', 'sentiment_label']).size().reset_index(name='count')
all_links = pd.concat([
    links_c_ug.rename(columns={'Q11': 'source', 'User_Group': 'target'}),
    links_ug_s.rename(columns={'User_Group': 'source', 'sentiment_label': 'target'})
], ignore_index=True)
label_indices = {label: i for i, label in enumerate(all_labels)}
all_links['source_idx'] = all_links['source'].map(label_indices)
all_links['target_idx'] = all_links['target'].map(label_indices)

# Create  Plotly Sankey figure
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 25,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = all_labels,
      color = all_colors
    ),
    link = dict(
      source = all_links['source_idx'],
      target = all_links['target_idx'],
      value = all_links['count'],
      color = '#EEEEEE'
  ))])

# 5. Update layout with bigger font
fig.update_layout(title_text="WSU Student Sentiment Flow: College -> Type of Use -> Sentiment", 
                  font_size=14,
                  font_family="Arial")

fig.write_html("level_5_sankey_diagram_FINAL_v2.html")
print("Saved 'level_5_sankey_diagram_FINAL_v2.html'")

print("\n--- ALL ANALYSIS COMPLETE ---")