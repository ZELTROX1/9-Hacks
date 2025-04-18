import streamlit as st
import requests
import json
import os
from PIL import Image
import io
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Define the API base URL
API_BASE_URL = "http://localhost:8000"

# App title and description
st.title("🏥 Medical AI Assistant")
st.markdown("""
This application provides access to medical AI services and document query capabilities.
Upload documents to create searchable knowledge bases and get intelligent medical insights.
""")

# Side navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a function", [
    "General Chat",
    "Medical Analysis",
    "Vector Database Management",
    "Document Query"
])

# Helper functions
def get_vector_dbs():
    """Fetch list of vector databases from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/list-vectordbs")
        if response.status_code == 200:
            return response.json().get("databases", [])
        else:
            st.error(f"Error fetching databases: {response.text}")
            return []
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return []

def display_query_results(response_data):
    """Display the results of a vector database query"""
    if "answer" in response_data:
        st.subheader("AI Response")
        st.write(response_data["answer"])
        
        st.subheader("Source Documents")
        for i, doc in enumerate(response_data.get("source_documents", [])):
            with st.expander(f"Document {i+1}"):
                st.write(doc["content"])
                
                # Display metadata if available
                if doc.get("metadata"):
                    st.write("**Metadata:**")
                    for key, value in doc["metadata"].items():
                        st.write(f"- {key}: {value}")

# ------ General Chat Page ------
if page == "General Chat":
    st.header("💬 General Chat")
    st.write("Ask any general questions to the AI assistant.")
    
    user_query = st.text_area("Your question:", height=100)
    
    if st.button("Send"):
        if user_query:
            with st.spinner("Processing..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/chat",
                        data={"query": user_query}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Response received")
                        
                        st.subheader("AI Response")
                        st.write(result["response"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Please enter a question.")

# ------ Medical Analysis Page ------
elif page == "Medical Analysis":
    st.header("🩺 Medical Analysis")
    st.write("Get medical insights from our specialized AI agent.")
    
    user_query = st.text_area("Medical question or describe symptoms:", height=100)
    
    # File upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Medical Documents")
        uploaded_docs = st.file_uploader(
            "Upload PDF or text files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="med_docs"
        )
    
    with col2:
        st.subheader("Upload X-ray Images")
        uploaded_xrays = st.file_uploader(
            "Upload X-ray images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="xrays"
        )
    
    if st.button("Analyze"):
        if user_query:
            with st.spinner("Analyzing medical data..."):
                try:
                    # Prepare files for upload
                    files = []
                    
                    # Add documents
                    if uploaded_docs:
                        for doc in uploaded_docs:
                            files.append(("documents", (doc.name, doc.getvalue(), doc.type)))
                    
                    # Add X-ray images
                    if uploaded_xrays:
                        for img in uploaded_xrays:
                            files.append(("xray_images", (img.name, img.getvalue(), img.type)))
                    
                    # Send request
                    response = requests.post(
                        f"{API_BASE_URL}/api/medical",
                        data={"query": user_query},
                        files=files
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("Analysis complete")
                        
                        st.subheader("Medical Analysis")
                        st.write(result["analysis"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Please enter a medical question or describe symptoms.")

# ------ Vector Database Management Page ------
elif page == "Vector Database Management":
    st.header("🗄️ Vector Database Management")
    
    # Tabs for different operations
    tab1, tab2, tab3 = st.tabs(["Create Database", "List Databases", "Delete Database"])
    
    # Create Database Tab
    with tab1:
        st.subheader("Create New Vector Database")
        
        db_name = st.text_input("Database Name (optional):", 
                               placeholder="Leave blank for auto-generated ID")
        
        uploaded_files = st.file_uploader(
            "Upload documents to create vector database",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="create_db_files"
        )
        
        if st.button("Create Database"):
            if uploaded_files:
                with st.spinner("Creating vector database..."):
                    try:
                        # Prepare files for upload
                        files = [("documents", (file.name, file.getvalue(), file.type)) 
                                for file in uploaded_files]
                        
                        # Send request
                        response = requests.post(
                            f"{API_BASE_URL}/api/create-vectordb",
                            data={"db_name": db_name} if db_name else {},
                            files=files
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Database created with ID: {result['db_id']}")
                            st.write(result["message"])
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            else:
                st.warning("Please upload at least one document.")
    
    # List Databases Tab
    with tab2:
        st.subheader("Available Vector Databases")
        
        if st.button("Refresh List"):
            st.session_state.dbs = get_vector_dbs()
        
        # Initialize session state if needed
        if "dbs" not in st.session_state:
            st.session_state.dbs = get_vector_dbs()
        
        if st.session_state.dbs:
            # Create a DataFrame for better display
            df = pd.DataFrame({"Database ID": st.session_state.dbs})
            st.table(df)
        else:
            st.info("No vector databases found. Create one first.")
    
    # Delete Database Tab
    with tab3:
        st.subheader("Delete Vector Database")
        
        # Get latest list of DBs for selection
        db_list = get_vector_dbs()
        
        if db_list:
            db_to_delete = st.selectbox("Select database to delete:", db_list)
            
            if st.button("Delete Database", type="primary", use_container_width=True):
                with st.spinner("Deleting database..."):
                    try:
                        response = requests.delete(f"{API_BASE_URL}/api/delete-vectordb/{db_to_delete}")
                        
                        if response.status_code == 200:
                            st.success(f"Database '{db_to_delete}' deleted successfully")
                            # Update the session state
                            if "dbs" in st.session_state:
                                st.session_state.dbs = [db for db in st.session_state.dbs if db != db_to_delete]
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
        else:
            st.info("No vector databases available to delete.")

# ------ Document Query Page ------
elif page == "Document Query":
    st.header("🔍 Query Documents")
    st.write("Search through your vector databases to find information.")
    
    # Get all available databases
    db_list = get_vector_dbs()
    
    if db_list:
        selected_db = st.selectbox("Select a vector database to query:", db_list)
        
        user_query = st.text_area("Your question:", height=100)
        
        if st.button("Search"):
            if user_query:
                with st.spinner("Searching documents..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/query-vectordb/{selected_db}",
                            data={"query": user_query}
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Search complete")
                            
                            # Display results
                            display_query_results(result)
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            else:
                st.warning("Please enter a question.")
    else:
        st.warning("No vector databases available. Please create one first.")

# Footer
st.markdown("---")
