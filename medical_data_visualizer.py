import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def preprocess_data():
    # Load data
    df = pd.read_csv('medical_examination.csv')
    
    # Add overweight column
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
    df['overweight'] = (df['BMI'] > 25).astype(int)
    
    # Normalize data
    df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
    df['gluc'] = (df['gluc'] > 1).astype(int)
    
    return df

def draw_cat_plot():
    # Preprocess the data
    df = preprocess_data()
    
    # Create a DataFrame for the categorical plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Draw the categorical plot
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='bar').fig
    plt.show()
    return fig

def draw_heat_map():
    # Preprocess the data
    df = preprocess_data()
    
    # Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Create the correlation matrix
    corr = df_heat.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', center=0)
    plt.show()
    
    return fig
