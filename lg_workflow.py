import json
import random

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, TypedDict, Any, Optional, List, Literal
from langgraph.graph import START, END, StateGraph
from constant import EXCLUDED_KEYS

from dotenv import load_dotenv
import logging
from logging_config import setup_logging

from langchain_tools import (
    get_patient_data_by_id,
    perform_similarity_search,
    calculate_risk_api,
    process_claim_details_and_decide_claim_status,
    provide_additional_claim_response_using_pubmed_and_guideline_api
)

load_dotenv()
setup_logging()
logger = logging.getLogger("langgraph_workflow")


class AgentState(TypedDict):
    patient_id: str
    patient_medical_details: Dict[str, Any]
    similarity_search_result: Optional[List]
    pubmed_search_result: Optional[List]
    latest_guideline_in_oncology: Optional[str]
    risk_score: float
    llm_suggested_claim_status: Literal["approve", "reject", "human review"]
    llm_suggested_claim_message: Optional[str]
    llm_provided_detailed_summary: Optional[str]
    detailed_summary_provided: Optional[bool]
    human_review_for_claim: Optional[Dict]
    human_review_feedback: Literal[
        "require_additional_information", "approve", "reject"
    ]
    claim_status: Literal["approve", "reject"]
    conversation_message: Optional[List] | List
    final_message: Optional[str]
    current_message: Optional[str]


def pull_patient_details(state: AgentState):
    """This agent is used to pull the patient details."""
    logger.info("start: parse_patient_details_agent.")
    try:
        current_message = state.get("current_message", None)
        conversation_message = state.get("conversation_message")
        if not current_message:
            raise ValueError(
                "There is no message provided to process. Aborting operation..!!"
            )
        patient_id = state.get("patient_id", None)
        if patient_id:
            patient_medical_data = get_patient_data_by_id.invoke(patient_id)
        else:
            raise ValueError(
                "No patient ID provided to process. Aborting operation..!!"
            )

        conversation_message.append(
            SystemMessage(
                f"pull_patient_details response: {json.dumps(patient_medical_data)}"
            )
        )
        logger.info("end: parse_patient_details_agent.")
        return {
            "current_message": json.dumps(patient_medical_data),
            "patient_medical_details": patient_medical_data,
            "conversation_message": conversation_message,
        }
    except ValueError:
        logger.info("end: parse_patient_details_agent.")
        raise ValueError()
    except Exception as e:
        logger.info("end: parse_patient_details_agent.")
        raise Exception(f"Exception while processing the request. {e}")


def perform_similarity_search_based_upon_patient_details(state: AgentState):
    """This agent is used to perform similarity search for given patient data."""
    logger.info("start: perform_similarity_search_agent")
    try:
        current_message = state.get("current_message", None)
        conversation_message = state.get("conversation_message", None)
        patient_medical_details = state.get("patient_medical_details", None)
        if not current_message:
            raise ValueError(
                "There is no message provided to process. Aborting operation..!!"
            )
        if patient_medical_details:
            patient_case_text = patient_medical_details.get("case_text")
            filter_blocks = []
            patient_keys = list(patient_medical_details.keys())
            for _key in patient_keys:
                if _key not in EXCLUDED_KEYS:
                    filter_blocks.append({_key: patient_medical_details[_key]})
            similar_search_content = perform_similarity_search.invoke(patient_case_text)
        else:
            raise ValueError(
                "No patient medical history details found. Aborting operation..!!"
            )
        conversation_message.append(
            SystemMessage(f"similar_search_content: {similar_search_content}")
        )
        current_message = json.dumps(similar_search_content)
        logger.info("end: perform_similarity_search_agent")
        return {
            "current_message": current_message,
            "conversation_message": conversation_message,
            "similarity_search_result": similar_search_content,
        }
    except ValueError:
        logger.info("end: perform_similarity_search_agent")
        raise ValueError()
    except Exception as e:
        logger.info("end: perform_similarity_search_agent")
        raise Exception(f"Exception while processing the request. {e}")


def calculate_risk_score_for_patient(state: AgentState):
    """This agent is used to calculate the risk score for given patient."""
    logger.info("start: calculate_risk_score_agent")
    try:
        current_message = state.get("current_message", None)
        conversation_message = state.get("conversation_message", None)
        patient_medical_details = state.get("patient_medical_details", None)
        if not current_message:
            raise ValueError(
                "There is no message provided to process. Aborting operation..!!"
            )
        if patient_medical_details:
            labs_data = []
            patient_keys = list(patient_medical_details.keys())
            for _key in patient_keys:
                if _key not in EXCLUDED_KEYS:
                    labs_data.append(f"{_key}: {patient_medical_details[_key]}")
            risk_score = calculate_risk_api.invoke(json.dumps(labs_data))
            if not risk_score:
                raise Exception("Error while calculating the risk score")
        if not current_message:
            raise ValueError(
                "There is no message provided to process. Aborting operation..!!"
            )
        conversation_message.append(SystemMessage(f"risk_score: {risk_score}"))
        current_message = json.dumps(risk_score)
        logger.info("end: calculate_risk_score_agent")
        return {
            "current_message": current_message,
            "conversation_message": conversation_message,
            "risk_score": risk_score,
        }
    except ValueError:
        logger.info("end: calculate_risk_score_agent")
        raise ValueError()
    except Exception as e:
        logger.info("end: calculate_risk_score_agent")
        raise Exception(f"Exception while processing the request. {e}")


def reasoning_and_decision_agent(state: AgentState):
    """This agent uses LLM to get the approval/rejection/human-in-loop condition for claim processing part."""
    logger.info("start: reasoning_and_decision_agent")
    try:
        current_message = state.get("current_message", None)
        conversation_message = state.get("conversation_message", None)
        patient_medical_details = state.get("patient_medical_details", None)
        similarity_search_result = state.get("similarity_search_result", None)
        risk_score = state.get("risk_score", None)
        if not current_message:
            raise ValueError(
                "There is no message provided to process. Aborting operation..!!"
            )
        dynamic_patient_dict = {
            "risk_score": risk_score,
            "similarity_search_result": similarity_search_result,
            "patient_medical_history": patient_medical_details,
        }
        claim_status_dict = process_claim_details_and_decide_claim_status.invoke(json.dumps(dynamic_patient_dict))
        logger.info("*"*20)
        logger.info(f"LLM suggested claim status: {claim_status_dict}")
        llm_suggested_claim_status = claim_status_dict["claim_status"].lower()
        llm_suggested_claim_message = claim_status_dict["summary"].lower()

        conversation_message.append(
            SystemMessage(
                f"llm_suggested_claim_status: {llm_suggested_claim_status},"
                f"llm_suggested_claim_message: {llm_suggested_claim_message}"
            )
        )
        current_message = json.dumps(claim_status_dict)
        logger.info("end: reasoning_and_decision_agent")
        return {
            "current_message": current_message,
            "conversation_message": conversation_message,
            "llm_suggested_claim_status": llm_suggested_claim_status,
            "llm_suggested_claim_message": llm_suggested_claim_message,
        }
    except ValueError:
        logger.info("end: reasoning_and_decision_agent")
        raise ValueError()
    except Exception as e:
        logger.info("end: reasoning_and_decision_agent")
        raise Exception(f"Exception while processing the request. {e}")


def additional_information_provider_with_pubmed_and_guideline_agent(state: AgentState):
    """This agent uses LLM, pubmed and guideline api to provide additional information for claim processing part."""
    logger.info("start: additional_information_provider_agent")
    try:
        current_message = state.get("current_message", None)
        conversation_message = state.get("conversation_message", [])
        patient_medical_details = state.get("patient_medical_details", None)
        similarity_search_result = state.get("similarity_search_result", None)
        risk_score = state.get("risk_score", None)
        if not current_message:
            raise ValueError(
                "There is no message provided to process. Aborting operation..!!"
            )
        if not conversation_message:
            conversation_message = []
        query_topic = "lung cancer immunotherapy"
        # pubmed_search_content = search_pubmed_api.invoke(query_topic)
        pubmed_search_content = json.dumps(
            [f"I'm a mock content of pubmed. Error is coming from the pubmed api."]
        )
        conversation_message.append(
            SystemMessage(f"pubmed_search_result: {pubmed_search_content}")
        )
        # guideline_api_search_content = fetch_guideline_api.invoke(query_topic)
        guideline_api_search_content = f"I'm a mock content of guideline_api. Error is coming from the guideline api."
        conversation_message.append(
            SystemMessage(
                f"latest_guideline_in_oncology: {guideline_api_search_content}"
            )
        )
        dynamic_patient_dict = {
            "risk_score": risk_score,
            "pubmed_api_result": pubmed_search_content,
            "guideline_api_result": guideline_api_search_content,
            "similarity_search_result": similarity_search_result,
            "patient_medical_history": patient_medical_details,
        }
        claim_status_dict = (
            provide_additional_claim_response_using_pubmed_and_guideline_api.invoke(
                json.dumps(dynamic_patient_dict)
            )
        )
        logger.info(f"ADDITIONAL INFORMATION PROVIDER AGENT provided details as below.\n{claim_status_dict}")
        llm_provided_detailed_summary = claim_status_dict["detailed_summary"].lower()
        conversation_message.append(
            SystemMessage(
                f"llm_provided_detailed_summary: {llm_provided_detailed_summary} and marked "
                f"human_review_feedback flag as None"
            )
        )
        current_message = json.dumps(claim_status_dict)
        logger.info("end: additional_information_provider_agent")
        return {
            "current_message": current_message,
            "conversation_message": conversation_message,
            "pubmed_search_result": pubmed_search_content,
            "latest_guideline_in_oncology": guideline_api_search_content,
            "llm_provided_detailed_summary": llm_provided_detailed_summary,
            "detailed_summary_provided": True,
            "human_review_feedback": None,
        }
    except ValueError:
        logger.info("end: additional_information_provider_agent")
        raise ValueError()
    except Exception as e:
        logger.info("end: additional_information_provider_agent")
        raise Exception(f"Exception while processing the request. {e}")


def conclude_claim_agent(state: AgentState):
    """This agent will approve the claim without additional information requested."""
    logger.info("start: conclude_claim_agent")
    human_feedback = state.get("human_review_feedback", None)
    llm_suggested_claim_status = state.get("llm_suggested_claim_status")
    conversation_message = state.get("conversation_message", [])
    if not conversation_message:
        conversation_message = []
    if llm_suggested_claim_status == "reject" and human_feedback == "not_initiated":
        return {
            "claim_status": f"{llm_suggested_claim_status}",
            "final_message": f"agents rejects the claim",
            "current_message": f"Claim for patient: {state.get('patient_id')} has been rejected",
            "conversation_message": conversation_message.append(
                SystemMessage(
                    f"Claim for patient: {state.get('patient_id')} has been rejected"
                )
            ),
        }
    elif llm_suggested_claim_status == "approve" and human_feedback == "not_initiated":
        return {
            "claim_status": f"{llm_suggested_claim_status}",
            "final_message": f"agents approved the claim",
            "current_message": f"Claim for patient: {state.get('patient_id')} has been approved",
            "conversation_message": conversation_message.append(
                SystemMessage(
                    f"Claim for patient: {state.get('patient_id')} has been approved"
                )
            ),
        }
    elif human_feedback.startswith("approve"):
        human_review_message = "Claim approved"
        human_review_status = "approved"
    elif human_feedback.startswith("reject"):
        human_review_message = "Claim rejected"
        human_review_status = "rejected"
    message = f"Claim for patient: {state.get('patient_id')} has been {human_feedback}"
    logger.info(message)
    logger.info("end: conclude_claim_agent")
    return {
        "claim_status": f"{human_review_status}",
        "final_message": f"{human_review_message}",
        "current_message": message,
        "conversation_message": conversation_message.append(SystemMessage(message)),
    }


def involve_human_review_agent(state: AgentState):
    """This agent is used to pause the graph for getting human review and resume again with review tasks."""
    logger.info("start: involve_human_review_agent")
    current_message = state.get("current_message", None)
    conversation_message = state.get("conversation_message", None)
    patient_medical_details = state.get("patient_medical_details", None)
    risk_score = state.get("risk_score", None)
    llm_suggested_claim_status = state.get("llm_suggested_claim_status")
    llm_suggested_claim_message = state.get("llm_suggested_claim_message")
    detailed_summary_provided = state.get("detailed_summary_provided")

    if not current_message:
        raise ValueError(
            "There is no message provided to process. Aborting operation..!!"
        )
    if detailed_summary_provided:
        logger.info(
            "human review still require additional information before giving claim status."
        )
        llm_provided_detailed_summary = state.get("llm_provided_detailed_summary")
        human_review_content = {
            "question": "Do you want to approve or require additional "
            "information for below claim details ?",
            "claim": f"""patient medical history: {json.dumps(patient_medical_details)} 
            showing risk score as: {json.dumps(risk_score)}
                    along with llm suggest claim message as: {json.dumps(llm_suggested_claim_message)} 
                    and  claim status as : {json.dumps(llm_suggested_claim_status)}. 
                    Please consider recent advancement in oncology: {json.dumps(llm_provided_detailed_summary)} 
                    Require your review to get claim decision.""",
        }
        # human_feedback = interrupt(human_review_content)
        updates = {
            # "human_review_feedback": f"{human_feedback}",
            "current_message": f"{json.dumps(human_review_content)}",
            "conversation_message": conversation_message.append(
                [
                    SystemMessage(
                        f"{human_review_content}",
                    )
                ]
            ),
        }
        logger.info("end: involve_human_review_agent")
        return updates
    else:
        human_review_content = {
            "question": "Do you want to approve or require additional "
            "information for below claim details ?",
            "claim": f"""patient medical history: {json.dumps(patient_medical_details)} 
                    showing risk score as: {json.dumps(risk_score)}
                            along with llm suggest claim message as: {json.dumps(llm_suggested_claim_message)} 
                            and  claim status as : {json.dumps(llm_suggested_claim_status)}. 
                            Require your review to get claim decision.""",
        }
        updates = {
            "current_message": f"{json.dumps(human_review_content)}",
            "conversation_message": conversation_message.append(
                [
                    SystemMessage(
                        f"{human_review_content}",
                    )
                ]
            ),
        }
        logger.info("end: involve_human_review_agent")
        return updates


def human_review_for_claim_decision_support_system(state):
    logger.info("start: human_review_for_claim_decision_support_system")
    if state["llm_suggested_claim_status"] == "approve":
        logger.info("end: human_review_for_claim_decision_support_system")
        return "involve_human_review_agent"
    elif state["llm_suggested_claim_status"] == "human review":
        logger.info("end: human_review_for_claim_decision_support_system")
        return "involve_human_review_agent"
    elif state["llm_suggested_claim_status"] == "reject":
        logger.info("end: human_review_for_claim_decision_support_system")
        return "involve_human_review_agent"


def route_human_review(state):
    feedback = state.get("human_review_feedback")
    logger.info(f"start: route_human_review with human_review_feedback: {feedback}")
    if feedback == "require_additional_information":
        logger.info("end: route_human_review")
        return "additional_information_provider_agent"
    else:
        logger.info("end: route_human_review")
        return "conclude_claim_agent"


def compile_graph():
    logger.info("start: compile_graph")
    memory_saver = MemorySaver()
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("parse_patient_details_agent", pull_patient_details)
    graph_builder.add_node(
        "perform_similarity_search_agent",
        perform_similarity_search_based_upon_patient_details,
    )
    graph_builder.add_node(
        "calculate_risk_score_agent", calculate_risk_score_for_patient
    )
    graph_builder.add_node("reasoning_and_decision_agent", reasoning_and_decision_agent)
    graph_builder.add_node("involve_human_review_agent", involve_human_review_agent)
    graph_builder.add_node(
        "additional_information_provider_agent",
        additional_information_provider_with_pubmed_and_guideline_agent,
    )
    graph_builder.add_node("conclude_claim_agent", conclude_claim_agent)
    graph_builder.add_edge(START, "parse_patient_details_agent")
    graph_builder.add_edge(
        "parse_patient_details_agent", "perform_similarity_search_agent"
    )
    graph_builder.add_edge(
        "perform_similarity_search_agent",
        "calculate_risk_score_agent",
    )
    graph_builder.add_edge("calculate_risk_score_agent", "reasoning_and_decision_agent")
    graph_builder.add_conditional_edges(
        "reasoning_and_decision_agent",
        human_review_for_claim_decision_support_system,
        {
            "involve_human_review_agent": "involve_human_review_agent",
            "conclude_claim_agent": "conclude_claim_agent",
        },
    )
    graph_builder.add_conditional_edges(
        "involve_human_review_agent",
        route_human_review,
        {
            "conclude_claim_agent": "conclude_claim_agent",
            "additional_information_provider_agent": "additional_information_provider_agent",
        },
    )
    graph_builder.add_edge(
        "additional_information_provider_agent", "involve_human_review_agent"
    )
    graph_builder.add_edge("conclude_claim_agent", END)
    graph_app = graph_builder.compile(
        checkpointer=memory_saver, interrupt_after=["involve_human_review_agent"]
    )
    logger.info("end: compile_graph")
    return graph_app, memory_saver
