import streamlit as st
import pickle
import requests
import json
import pandas as pd

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Claim Processing Dashboard")
API_BASE_URL = "http://localhost:8000"
PROCESS_CLAIM_URL = f"{API_BASE_URL}/process-claim"
FEEDBACK_URL = f"{API_BASE_URL}/process-claim-with-human-feedback"


# --- Helper Functions ---

def load_patient_data():
    """Loads patient data from the pickle file."""
    try:
        with open("patient_data.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: 'patient_data.pkl' not found. Please create it first.")
        return []


def safe_json_loads(s):
    """Safely parse a JSON string, returning a dictionary or None on failure."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def initialize_session_state():
    """Initializes the session state for the application."""
    if 'claims_data' not in st.session_state:
        patients = load_patient_data()
        st.session_state.claims_data = {
            p['patient_id']: {'status': 'Unprocessed', 'data': p} for p in patients
        }
    if 'selected_patient_id' not in st.session_state:
        st.session_state.selected_patient_id = None
    if 'agent_response' not in st.session_state:
        st.session_state.agent_response = None


def process_claim_request(patient_id):
    """Handles the initial request to process a claim."""
    st.session_state.selected_patient_id = patient_id
    st.session_state.agent_response = None  # Clear previous response
    try:
        with st.spinner(f"Agent is analyzing claim for {patient_id}..."):
            response = requests.post(PROCESS_CLAIM_URL, json={"patient_id": patient_id}, timeout=60)
            response.raise_for_status()
            st.session_state.agent_response = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error while processing claim: {e}")
        st.session_state.selected_patient_id = None  # Reset on error


def submit_feedback_request(patient_id, feedback):
    """Handles the submission of human feedback."""
    try:
        with st.spinner(f"Submitting feedback for {patient_id}..."):
            payload = {"patient_id": patient_id, "human_review_feedback": feedback}
            response = requests.post(FEEDBACK_URL, json=payload, timeout=60)
            response.raise_for_status()
            agent_response = response.json()

            # Check if the claim is concluded
            if agent_response.get("claim_status") in ["approved", "rejected"]:
                final_status = agent_response["claim_status"].capitalize()
                st.session_state.claims_data[patient_id]['status'] = final_status
                if final_status == "Approved":
                    st.success(f"✅ Claim for {patient_id} has been Approved.")
                else:
                    st.error(f"❌ Claim for {patient_id} has been Rejected.")
                # Reset state to go back to the main dashboard
                st.session_state.selected_patient_id = None
                st.session_state.agent_response = None
            else:
                # Update the response to continue the loop
                st.session_state.agent_response = agent_response
                st.info("Additional information has been provided. Please review again.")
    except requests.exceptions.RequestException as e:
        st.error(f"API Error while submitting feedback: {e}")


# --- UI Rendering Functions ---

def render_claim_review_ui(patient_id, response_data):
    """Renders the detailed UI for reviewing a single claim."""
    st.header(f"Reviewing Claim for Patient: `{patient_id}`")

    # Parse the nested JSON from the 'current_message' field
    current_message_str = response_data.get("current_message", "{}")
    review_details = safe_json_loads(current_message_str)

    if not review_details:
        st.warning("Could not parse agent's current message. Displaying raw data.")
        st.json(response_data)
        return

    st.subheader("Agent's Assessment")
    st.markdown(f"> {review_details.get('question', 'No question provided.')}")

    with st.container(border=True):
        claim_info = review_details.get('claim', 'No claim details available.')
        # Split the string to format it nicely
        for line in claim_info.split('\n'):
            st.text(line.strip())

    st.subheader("Your Decision")
    feedback = st.radio(
        "Select your action for this claim:",
        options=["approve", "reject", "require_additional_information"],
        captions=["Final approval", "Final rejection", "Request more details from the agent"],
        key=f"feedback_{patient_id}",
        horizontal=True
    )

    if st.button("Submit Decision", type="primary"):
        submit_feedback_request(patient_id, feedback)
        st.rerun()


def render_dashboard():
    """Renders the main dashboard with three columns for claim statuses."""
    st.title("Insurance Claim Processing Dashboard")

    # Filter claims by status
    unprocessed = {k: v for k, v in st.session_state.claims_data.items() if v['status'] == 'Unprocessed'}
    approved = {k: v for k, v in st.session_state.claims_data.items() if v['status'] == 'Approved'}
    rejected = {k: v for k, v in st.session_state.claims_data.items() if v['status'] == 'Rejected'}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header(f"Unprocessed ({len(unprocessed)})")
        for patient_id, details in unprocessed.items():
            with st.container(border=True):
                st.subheader(f"Patient ID: {patient_id}")
                with st.expander("Show Patient Details"):
                    st.json(details['data'])
                if st.button(f"Process Claim for {patient_id}", key=f"btn_{patient_id}"):
                    process_claim_request(patient_id)
                    st.rerun()

    with col2:
        st.header(f"Approved ({len(approved)})")
        for patient_id, details in approved.items():
            with st.container(border=True):
                st.subheader(f"Patient ID: {patient_id}")
                st.success("This claim has been approved.")
                with st.expander("View Patient Details"):
                    st.json(details['data'])

    with col3:
        st.header(f"Rejected ({len(rejected)})")
        for patient_id, details in rejected.items():
            with st.container(border=True):
                st.subheader(f"Patient ID: {patient_id}")
                st.error("This claim has been rejected.")
                with st.expander("View Patient Details"):
                    st.json(details['data'])


# --- Main Application Logic ---

initialize_session_state()

if st.session_state.selected_patient_id:
    # If a patient is selected, show the detailed review UI
    if st.session_state.agent_response:
        render_claim_review_ui(st.session_state.selected_patient_id, st.session_state.agent_response)
    else:
        # This handles the case where API call is in progress or failed
        st.info("Loading agent analysis...")
else:
    # Otherwise, show the main dashboard
    render_dashboard()