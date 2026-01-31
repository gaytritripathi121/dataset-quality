import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from main import DatasetQualityAuditor
from utils.helpers import create_sample_data


# Page config
st.set_page_config(
    page_title="Dataset Quality Auditor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_download_link(file_path, link_text):
    """Generate download link for file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{Path(file_path).name}">{link_text}</a>'
    return href


def main():
    # Title
    st.title("Dataset Quality Auditor")
    st.markdown("### Automated ML-Readiness Assessment")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        audit_mode = st.radio(
            "Select Mode:",
            ["Upload Dataset", "Use Sample Data"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This tool performs comprehensive quality checks on datasets using:\n"
            "- Statistical profiling\n"
            "- ML anomaly detection\n"
            "- Health scoring (0-100)"
        )
    
    # Main content
    if audit_mode == "Upload Dataset":
        st.header("Upload Your Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV, Excel, or Parquet file",
            type=['csv', 'xlsx', 'parquet']
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Preview data
            st.subheader("Data Preview")
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(tmp_path)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(tmp_path)
                else:
                    df = pd.read_parquet(tmp_path)
                
                st.dataframe(df.head(10))
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Run audit button
                if st.button("Run Quality Audit", type="primary"):
                    run_audit(tmp_path)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    else:  # Sample Data
        st.header("Generate Sample Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples = st.number_input("Number of Samples", 100, 10000, 1000, 100)
        with col2:
            n_features = st.number_input("Number of Features", 3, 50, 10, 1)
        with col3:
            anomaly_ratio = st.slider("Anomaly Ratio", 0.0, 0.3, 0.1, 0.01)
        
        if st.button("Generate and Audit Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                df = create_sample_data(n_samples, n_features, anomaly_ratio)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    df.to_csv(tmp_file.name, index=False)
                    tmp_path = tmp_file.name
                
                st.success("Sample data generated")
                st.dataframe(df.head(10))
                
                # Run audit
                run_audit(tmp_path)


def run_audit(file_path):
    """Run audit and display results"""
    st.markdown("---")
    st.header("Audit Results")
    
    # Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize auditor
        status_text.text("Initializing auditor...")
        progress_bar.progress(10)
        
        auditor = DatasetQualityAuditor()
        
        # Run audit
        status_text.text("Running quality audit...")
        progress_bar.progress(30)
        
        results = auditor.audit_dataset(file_path)
        
        progress_bar.progress(100)
        status_text.text("Audit complete")
        
        # Display results
        display_results(results)
        
    except Exception as e:
        st.error(f"Audit failed: {str(e)}")


def display_results(results):
    """Display audit results"""
    scores = results['scores']
    
    # Overall score
    st.markdown("### Overall Health Score")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score", f"{scores['overall_score']}/100")
    
    with col2:
        st.metric("Rating", scores['rating'])
    
    with col3:
        st.info(scores['recommendation'])
    
    # Component scores
    st.markdown("### Component Scores")
    
    cols = st.columns(4)
    components = scores['component_scores']
    
    for i, (component, score) in enumerate(components.items()):
        with cols[i]:
            st.metric(component.capitalize(), f"{score}/25")
    
    # Visualizations
    st.markdown("### Visualizations")
    
    if results.get('plot_paths'):
        for plot_path in results['plot_paths']:
            if Path(plot_path).exists():
                st.image(str(plot_path), use_container_width=True)
    
    # Download reports
    st.markdown("### Download Reports")
    
    if results.get('report_paths'):
        col1, col2, col3 = st.columns(3)
        report_paths = results['report_paths']
        
        if 'json_report' in report_paths and Path(report_paths['json_report']).exists():
            with col1:
                st.markdown(
                    get_download_link(report_paths['json_report'], "JSON Report"),
                    unsafe_allow_html=True
                )
        
        if 'text_summary' in report_paths and Path(report_paths['text_summary']).exists():
            with col2:
                st.markdown(
                    get_download_link(report_paths['text_summary'], "Text Summary"),
                    unsafe_allow_html=True
                )
        
        if 'html_report' in report_paths and Path(report_paths['html_report']).exists():
            with col3:
                st.markdown(
                    get_download_link(report_paths['html_report'], "HTML Report"),
                    unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()
