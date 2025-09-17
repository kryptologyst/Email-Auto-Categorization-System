"""
Modern Streamlit UI for Email Auto-Categorization System
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.database import EmailDatabase
from src.models import EmailClassifier, EnsembleEmailClassifier
from src.visualization import EmailVisualization
import os
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Email Auto-Categorization System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load email data from database."""
    db = EmailDatabase()
    if not os.path.exists("emails.db"):
        db.add_sample_data()
    return db.get_all_emails()

@st.cache_resource
def load_model(model_type="ensemble"):
    """Load and train the email classification model."""
    db = EmailDatabase()
    if not os.path.exists("emails.db"):
        db.add_sample_data()
    
    df = db.get_training_emails()
    
    if model_type == "ensemble":
        model = EnsembleEmailClassifier()
    else:
        model = EmailClassifier(model_type)
    
    with st.spinner(f"Training {model_type} model..."):
        results = model.train(df)
    
    return model, results

def main():
    # Header
    st.markdown('<h1 class="main-header">📧 Email Auto-Categorization System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔍 Classify Email", "📊 Analytics", "🗄️ Database", "⚙️ Model Training"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Classify Email":
        show_classification_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "🗄️ Database":
        show_database_page()
    elif page == "⚙️ Model Training":
        show_training_page()

def show_home_page():
    """Display the home page with overview."""
    st.markdown("## Welcome to the Email Auto-Categorization System")
    
    col1, col2, col3 = st.columns(3)
    
    # Load data for metrics
    df = load_data()
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Emails", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Categories", df['category'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Emails", len(df[df['is_training'] == 1]))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick overview chart
    st.markdown("### Email Distribution by Category")
    fig = px.pie(df, names='category', title="Email Categories")
    st.plotly_chart(fig, use_container_width=True)
    
    # Features section
    st.markdown("### 🚀 Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Advanced ML Models**
        - Multiple algorithms (Logistic Regression, Random Forest, SVM)
        - Ensemble learning for better accuracy
        - Cross-validation for robust evaluation
        
        **📊 Rich Analytics**
        - Interactive visualizations
        - Performance metrics
        - Feature importance analysis
        """)
    
    with col2:
        st.markdown("""
        **🗄️ Database Integration**
        - SQLite database for email storage
        - Sample data generation
        - Easy data management
        
        **🎨 Modern UI**
        - Streamlit-based interface
        - Real-time predictions
        - Responsive design
        """)

def show_classification_page():
    """Display the email classification page."""
    st.markdown("## 🔍 Classify Your Email")
    
    # Model selection
    model_type = st.selectbox(
        "Choose Model Type",
        ["ensemble", "logistic_regression", "random_forest", "gradient_boosting", "svm"]
    )
    
    # Load model
    try:
        model, _ = load_model(model_type)
        st.success(f"✅ {model_type.replace('_', ' ').title()} model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    # Input form
    with st.form("email_form"):
        st.markdown("### Enter Email Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            subject = st.text_input("Email Subject", placeholder="Enter email subject...")
            sender = st.text_input("Sender Email", placeholder="sender@example.com")
        
        with col2:
            recipient = st.text_input("Recipient Email", placeholder="recipient@example.com")
        
        body = st.text_area("Email Body", height=200, placeholder="Enter email content...")
        
        submitted = st.form_submit_button("🔍 Classify Email", use_container_width=True)
    
    if submitted and (subject or body):
        with st.spinner("Analyzing email..."):
            try:
                # Make prediction
                category, confidence = model.predict(body, subject)
                
                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>📧 Prediction Result</h2>
                    <h3>Category: {category}</h3>
                    <h4>Confidence: {confidence:.2%}</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to save to database
                if st.button("💾 Save to Database"):
                    db = EmailDatabase()
                    db.add_email(subject, body, sender, recipient, category, confidence, False)
                    st.success("Email saved to database!")
                    
            except Exception as e:
                st.error(f"❌ Error during classification: {str(e)}")
    
    elif submitted:
        st.warning("⚠️ Please enter at least a subject or body text.")

def show_analytics_page():
    """Display analytics and visualizations."""
    st.markdown("## 📊 Email Analytics Dashboard")
    
    df = load_data()
    viz = EmailVisualization()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", len(df))
    with col2:
        st.metric("Categories", df['category'].nunique())
    with col3:
        avg_confidence = df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    with col4:
        recent_emails = len(df[df['created_at'] >= df['created_at'].max()])
        st.metric("Recent Emails", recent_emails)
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "🔤 Word Clouds", "📈 Performance", "📅 Timeline"])
    
    with tab1:
        st.markdown("### Category Distribution")
        fig = viz.plot_category_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Email Statistics by Category")
        category_stats = df.groupby('category').agg({
            'subject': 'count',
            'confidence': 'mean',
            'created_at': 'max'
        }).round(3)
        category_stats.columns = ['Count', 'Avg Confidence', 'Latest Email']
        st.dataframe(category_stats, use_container_width=True)
    
    with tab2:
        st.markdown("### Word Clouds by Category")
        categories = df['category'].unique()
        
        for category in categories:
            st.markdown(f"#### {category} Emails")
            try:
                fig = viz.create_wordcloud(df, category)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate word cloud for {category}: {str(e)}")
    
    with tab3:
        st.markdown("### Model Performance Comparison")
        
        if st.button("🔄 Run Model Comparison"):
            with st.spinner("Training and comparing models..."):
                models = ["logistic_regression", "random_forest", "gradient_boosting"]
                results = {}
                
                progress_bar = st.progress(0)
                
                for i, model_type in enumerate(models):
                    try:
                        model = EmailClassifier(model_type)
                        result = model.train(df[df['is_training'] == 1])
                        results[model_type] = result
                        progress_bar.progress((i + 1) / len(models))
                    except Exception as e:
                        st.error(f"Error training {model_type}: {str(e)}")
                
                if results:
                    fig = viz.plot_model_comparison(results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results table
                    st.markdown("### Detailed Results")
                    results_df = pd.DataFrame({
                        model: {
                            'Accuracy': results[model]['accuracy'],
                            'CV Mean': results[model]['cv_mean'],
                            'CV Std': results[model]['cv_std']
                        } for model in results.keys()
                    }).T
                    st.dataframe(results_df.round(4), use_container_width=True)
    
    with tab4:
        st.markdown("### Email Timeline")
        if 'created_at' in df.columns:
            try:
                fig = viz.plot_email_timeline(df)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate timeline: {str(e)}")
        else:
            st.info("Timeline data not available")

def show_database_page():
    """Display database management page."""
    st.markdown("## 🗄️ Database Management")
    
    db = EmailDatabase()
    df = load_data()
    
    # Database stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Training Records", len(df[df['is_training'] == 1]))
    with col3:
        st.metric("Test Records", len(df[df['is_training'] == 0]))
    
    # Data table
    st.markdown("### Email Database")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        category_filter = st.selectbox("Filter by Category", ["All"] + list(df['category'].unique()))
    
    with col2:
        training_filter = st.selectbox("Filter by Type", ["All", "Training", "Test"])
    
    # Apply filters
    filtered_df = df.copy()
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    if training_filter == "Training":
        filtered_df = filtered_df[filtered_df['is_training'] == 1]
    elif training_filter == "Test":
        filtered_df = filtered_df[filtered_df['is_training'] == 0]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Database operations
    st.markdown("### Database Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Sample Data"):
            with st.spinner("Adding sample data..."):
                db.add_sample_data()
                st.success("Sample data refreshed!")
                st.experimental_rerun()
    
    with col2:
        if st.button("📊 Show Category Stats"):
            stats = db.get_category_stats()
            st.dataframe(stats, use_container_width=True)

def show_training_page():
    """Display model training page."""
    st.markdown("## ⚙️ Model Training & Evaluation")
    
    df = load_data()
    training_df = df[df['is_training'] == 1]
    
    st.markdown(f"### Training Data: {len(training_df)} emails")
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["logistic_regression", "random_forest", "gradient_boosting", "svm", "ensemble"],
        help="Choose the machine learning algorithm to train"
    )
    
    # Training parameters
    with st.expander("⚙️ Advanced Parameters"):
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        random_state = st.number_input("Random State", 0, 100, 42)
    
    # Train model
    if st.button("🚀 Train Model", use_container_width=True):
        with st.spinner(f"Training {model_type} model..."):
            try:
                if model_type == "ensemble":
                    model = EnsembleEmailClassifier()
                else:
                    model = EmailClassifier(model_type)
                
                results = model.train(training_df)
                
                # Display results
                st.success("✅ Model trained successfully!")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Test Accuracy", f"{results['accuracy']:.3f}")
                
                with col2:
                    if 'cv_mean' in results:
                        st.metric("CV Mean", f"{results['cv_mean']:.3f}")
                
                with col3:
                    if 'cv_std' in results:
                        st.metric("CV Std", f"{results['cv_std']:.3f}")
                
                # Classification report
                if 'classification_report' in results:
                    st.markdown("### 📋 Classification Report")
                    report_df = pd.DataFrame(results['classification_report']).transpose()
                    st.dataframe(report_df.round(3), use_container_width=True)
                
                # Confusion matrix
                if 'confusion_matrix' in results:
                    st.markdown("### 🔄 Confusion Matrix")
                    viz = EmailVisualization()
                    # This would need actual y_true and y_pred for proper visualization
                    st.info("Confusion matrix visualization available in full training pipeline")
                
                # Save model option
                if st.button("💾 Save Model"):
                    model_path = f"models/{model_type}_model.joblib"
                    os.makedirs("models", exist_ok=True)
                    model.save_model(model_path)
                    st.success(f"Model saved to {model_path}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
    
    # Model comparison
    st.markdown("### 🏆 Model Comparison")
    
    if st.button("🔄 Compare All Models"):
        models_to_compare = ["logistic_regression", "random_forest", "gradient_boosting"]
        comparison_results = {}
        
        progress_bar = st.progress(0)
        
        for i, model_name in enumerate(models_to_compare):
            with st.spinner(f"Training {model_name}..."):
                try:
                    model = EmailClassifier(model_name)
                    result = model.train(training_df)
                    comparison_results[model_name] = result
                    progress_bar.progress((i + 1) / len(models_to_compare))
                except Exception as e:
                    st.error(f"Error training {model_name}: {str(e)}")
        
        if comparison_results:
            # Create comparison chart
            viz = EmailVisualization()
            fig = viz.plot_model_comparison(comparison_results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model recommendation
            best_model = max(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]['accuracy'])
            st.success(f"🏆 Best performing model: **{best_model}** "
                      f"(Accuracy: {comparison_results[best_model]['accuracy']:.3f})")

if __name__ == "__main__":
    main()
