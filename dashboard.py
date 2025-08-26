import streamlit as st
import json
import pickle
from langchain_core.messages import HumanMessage

# Import the necessary components from your existing agent workflow
# Ensure agent_workflow.py is in the same directory
# from agent_workflow import compile_graph
from execute_graph import compile_graph

# --- Page Configuration and State Initialization ---

st.set_page_config(
    page_title="Agentic Claims Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🩺 Agentic Insurance Claim Review Dashboard")


# Initialize the LangGraph app and memory saver once
@st.cache_resource
def load_agent_app():
    return compile_graph()


graph_app, memory_saver = load_agent_app()

# Initialize session state for claims and active claim
if 'claims' not in st.session_state:
    try:
        # Load patient data from the pickle file
        with open("patient_data.pkl", "rb") as f:
            patients = pickle.load(f)

        # Initialize claims in session state
        st.session_state.claims = {
            p['patient_id']: {'status': 'Unprocessed', 'data': p, 'state': None, 'history': []}
            for p in patients
        }
    except FileNotFoundError:
        st.error("Error: `patient_data.pkl` not found. Please ensure the patient data file exists.")
        st.stop()

if 'active_claim_id' not in st.session_state:
    st.session_state.active_claim_id = None


# --- Helper Functions for Agent Interaction ---

def start_claim_processing(claim_id):
    """Initiates the agentic workflow for a specific claim."""
    st.session_state.claims[claim_id]['status'] = 'Processing'
    st.session_state.claims[claim_id]['history'].append("▶️ Claim processing started.")

    config = {"configurable": {"thread_id": claim_id}}
    user_query = f"Please process claim for patient_id: {claim_id}"
    initial_state = {
        "patient_id": claim_id,
        "current_message": user_query,
        "conversation_message": [HumanMessage(user_query)],
        "human_review_feedback": "not_initiated"
    }

    # Run the graph until it's interrupted or finishes
    with st.spinner(f"Agent is analyzing claim {claim_id}..."):
        graph_app.invoke(initial_state, config)


def submit_human_feedback(claim_id, feedback):
    """Submits human feedback and resumes the workflow."""
    st.session_state.claims[claim_id]['history'].append(f"👤 Human feedback: **{feedback.replace('_', ' ').title()}**")
    config = {"configurable": {"thread_id": claim_id}}

    with st.spinner("Agent is incorporating your feedback..."):
        # Update the state with the human's feedback
        graph_app.update_state(config, {"human_review_feedback": feedback})
        # Resume the graph execution
        graph_app.invoke(None, config)


# --- UI Layout ---

col1, col2 = st.columns([1, 2])

# Column 1: Dashboard of all claims
with col1:
    st.header("Claims Queue")
    sorted_claims = sorted(st.session_state.claims.items())

    for claim_id, claim_info in sorted_claims:
        status = claim_info['status']
        btn_type = "primary" if status == "Pending Review" else "secondary"
        if st.button(f"**{claim_id}** - `{status}`", key=claim_id, use_container_width=True, type=btn_type):
            st.session_state.active_claim_id = claim_id
            st.rerun()

# Column 2: Details view for the active claim
with col2:
    st.header("Claim Details")
    active_id = st.session_state.active_claim_id

    if not active_id:
        st.info("Select a claim from the queue on the left to view its details.")
    else:
        claim_data = st.session_state.claims[active_id]['data']
        config = {"configurable": {"thread_id": active_id}}

        # Get the latest state from the memory saver
        latest_state_snapshot = memory_saver.get(config)
        agent_state = latest_state_snapshot.values if latest_state_snapshot else {}

        # Update our local status based on the agent's state
        if latest_state_snapshot.get("final_message"):
            st.session_state.claims[active_id]['status'] = f"{st.session_state.claims[active_id]['status']['decision']}"
        elif latest_state_snapshot and any(cmd.name == 'interrupt' for cmd in latest_state_snapshot.next):
            st.session_state.claims[active_id]['status'] = "Human Review"

        # Display basic patient details
        st.subheader(f"Patient ID: {claim_data['patient_id']}")
        with st.expander("Patient Medical History", expanded=False):
            st.json(claim_data)

        st.markdown("---")

        status = st.session_state.claims[active_id]['status']

        # --- Display Area for Agent Output and Human Interaction ---

        # Case 1: Claim is unprocessed
        if status == 'Unprocessed':
            if st.button("Start Claim Processing", type="primary"):
                start_claim_processing(active_id)
                st.rerun()

        # Case 2: Claim is concluded
        elif "Concluded" in status:
            final_status = agent_state.get("claim_status", "Unknown").title()
            final_message = agent_state.get("final_message", "No final message available.")
            if final_status == "Approved":
                st.success(f"**Status: Claim {final_status}**")
            else:
                st.error(f"**Status: Claim {final_status}**")
            st.write(final_message)

        # Case 3: Claim is processing or pending review
        else:
            # Show agent's findings
            if agent_state:
                st.subheader("Agent's Analysis 🤖")
                llm_summary = agent_state.get("llm_suggested_claim_message")
                llm_decision = agent_state.get("llm_suggested_claim_status", "N/A").title()
                risk_score = agent_state.get("risk_score")

                st.metric("Calculated Risk Score", f"{risk_score:.2f}")
                st.info(f"**Agent Suggestion:** `{llm_decision}`\n\n**Summary:** {llm_summary}")

                # Show additional info if it was requested and provided
                if agent_state.get("detailed_summary_provided"):
                    st.warning("**Additional Information Requested and Provided:**")
                    detailed_summary = agent_state.get("llm_provided_detailed_summary", "No details found.")
                    st.text_area("Recent Advancements in Oncology (from PubMed/Guideline APIs)", value=detailed_summary,
                                 height=200, disabled=True)

            # Show interaction controls if pending review
            if status == "Pending Review":
                st.markdown("---")
                st.subheader("Human Review Required 👤")
                st.write("The agent has paused for your decision. Please review the analysis and choose an action.")

                review_cols = st.columns(3)
                with review_cols[0]:
                    if st.button("✅ Approve Claim", use_container_width=True):
                        submit_human_feedback(active_id, "approve")
                        st.rerun()

                with review_cols[1]:
                    if st.button("❌ Reject Claim", use_container_width=True):
                        submit_human_feedback(active_id, "reject")
                        st.rerun()

                with review_cols[2]:
                    # Only show this button if additional info hasn't been provided yet
                    if not agent_state.get("detailed_summary_provided"):
                        if st.button("ℹ️ Request Additional Info", use_container_width=True):
                            submit_human_feedback(active_id, "require_additional_information")
                            st.rerun()

        # Display the history of actions for this claim
        st.markdown("---")
        with st.expander("Claim Processing History"):
            for entry in st.session_state.claims[active_id]['history']:
                st.markdown(f"- {entry}")