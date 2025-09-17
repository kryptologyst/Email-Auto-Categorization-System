"""
Advanced visualization module for email categorization analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class EmailVisualization:
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_category_distribution(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Create an interactive pie chart of email categories."""
        category_counts = df['category'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Email Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            title_font_size=20,
            font=dict(size=14),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str], save_path: str = None) -> plt.Figure:
        """Create an enhanced confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels, 
            yticklabels=labels,
            ax=ax,
            cbar_kws={'label': 'Number of Emails'}
        )
        
        ax.set_title('Confusion Matrix - Email Categorization', fontsize=16, pad=20)
        ax.set_xlabel('Predicted Category', fontsize=14)
        ax.set_ylabel('Actual Category', fontsize=14)
        
        # Add accuracy text
        accuracy = np.trace(cm) / np.sum(cm)
        ax.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.3f}', 
                transform=ax.transAxes, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict], save_path: str = None) -> go.Figure:
        """Compare performance of different models."""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        cv_means = [results[model]['cv_mean'] for model in models]
        cv_stds = [results[model]['cv_std'] for model in models]
        
        fig = go.Figure()
        
        # Add accuracy bars
        fig.add_trace(go.Bar(
            name='Test Accuracy',
            x=models,
            y=accuracies,
            marker_color='lightblue',
            text=[f'{acc:.3f}' for acc in accuracies],
            textposition='auto'
        ))
        
        # Add CV score bars with error bars
        fig.add_trace(go.Bar(
            name='CV Mean ± Std',
            x=models,
            y=cv_means,
            error_y=dict(type='data', array=cv_stds),
            marker_color='lightcoral',
            text=[f'{mean:.3f}±{std:.3f}' for mean, std in zip(cv_means, cv_stds)],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Models',
            yaxis_title='Accuracy',
            barmode='group',
            title_font_size=20,
            font=dict(size=14)
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_wordcloud(self, df: pd.DataFrame, category: str = None, 
                        save_path: str = None) -> plt.Figure:
        """Create word cloud for email content."""
        if category:
            text_data = df[df['category'] == category]
            title = f'Word Cloud - {category} Emails'
        else:
            text_data = df
            title = 'Word Cloud - All Emails'
        
        # Combine all text
        all_text = ' '.join(text_data['subject'].fillna('') + ' ' + text_data['body'].fillna(''))
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_feature_importance(self, model, vectorizer, top_n: int = 20, 
                              save_path: str = None) -> plt.Figure:
        """Plot feature importance for tree-based models."""
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model doesn't have feature_importances_ attribute")
        
        # Get feature names and importances
        feature_names = vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        
        # Get top features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_importances)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Most Important Features', fontsize=16)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_email_timeline(self, df: pd.DataFrame, save_path: str = None) -> go.Figure:
        """Plot email distribution over time."""
        if 'created_at' not in df.columns:
            raise ValueError("DataFrame must have 'created_at' column")
        
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = df['created_at'].dt.date
        
        # Count emails per day by category
        timeline_data = df.groupby(['date', 'category']).size().reset_index(name='count')
        
        fig = px.line(
            timeline_data, 
            x='date', 
            y='count', 
            color='category',
            title='Email Volume Over Time by Category',
            labels={'count': 'Number of Emails', 'date': 'Date'}
        )
        
        fig.update_layout(
            title_font_size=20,
            font=dict(size=14),
            xaxis_title='Date',
            yaxis_title='Number of Emails'
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def create_dashboard_summary(self, df: pd.DataFrame, model_results: Dict = None) -> go.Figure:
        """Create a comprehensive dashboard summary."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Category Distribution', 'Emails by Sender Domain', 
                          'Subject Length Distribution', 'Model Performance'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Category distribution (pie chart)
        category_counts = df['category'].value_counts()
        fig.add_trace(
            go.Pie(labels=category_counts.index, values=category_counts.values, name="Categories"),
            row=1, col=1
        )
        
        # Sender domains (bar chart)
        df['sender_domain'] = df['sender'].str.split('@').str[1]
        domain_counts = df['sender_domain'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=domain_counts.index, y=domain_counts.values, name="Domains"),
            row=1, col=2
        )
        
        # Subject length distribution
        df['subject_length'] = df['subject'].str.len()
        fig.add_trace(
            go.Histogram(x=df['subject_length'], name="Subject Length"),
            row=2, col=1
        )
        
        # Model performance (if available)
        if model_results:
            models = list(model_results.keys())
            accuracies = [model_results[model]['accuracy'] for model in models]
            fig.add_trace(
                go.Bar(x=models, y=accuracies, name="Accuracy"),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Email Categorization Dashboard",
            title_font_size=24,
            showlegend=False,
            height=800
        )
        
        return fig
