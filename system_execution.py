import pickle
import json
import requests
import re
import pandas as pd
from chromadb.utils import embedding_functions
import chromadb
import os
import random

from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Dict, TypedDict, Any, Optional, List, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from constant import (
    BIOMARKER_WEIGHTS,
    TOP_K,
    EXCLUDED_KEYS,
    PUBMED_API_URL,
    GUIDELINE_API_URL,
)
from prompt import (
    generate_prompt_with_similar_search_content_and_risk_score,
    generate_prompt_with_all_possible_parameters,
)
from dotenv import load_dotenv
import logging
from logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger("system execution")

class LLMResponseSchema(BaseModel):
    summary: str = Field(description="A concise summary behind claim status")
    claim_status: Literal["approve", "reject", "human review"] = Field(
        description="To provide claim status based upon provided literal value"
    )



class LLMResponseSchemaForDetailedSummay(BaseModel):
    detailed_summary: str = Field(
        description="A brief summary with recent "
        "advancements in oncology guided by PubMed and Guideline api"
    )


def get_patient_data() -> List:
    """
    This method use to provide the all possible patient details present in system.
    :return: List containing patient data.
    """
    with open("patient_data.pkl", "rb") as f:
        patients = pickle.load(f)
    logger.debug(f"patient data: {patients}")
    return patients


def parse_labs_with_flags(labs_list: str) -> Dict:
    metadata = {
        "pd_l1": 0,
        "pd_l1_available": False,
        "tmb": 0,
        "tmb_available": False,
        "msi_h": False,
        "msi_h_available": False,
        "progressive_disease": False,
        "progressive_disease_available": False,
        "prior_therapy_failure": False,
        "prior_therapy_failure_available": False,
        "mutations": [],
        "mutations_available": False,
    }

    # Iterate over labs list
    for lab in labs_list:
        l = lab.lower()

        if "pd-l1" in l:
            match = re.search(r"pd-l1:\s*(\d+)", lab, re.IGNORECASE)
            if match:
                metadata["pd_l1"] = int(match.group(1))
                metadata["pd_l1_available"] = True
            else:
                metadata["pd_l1"] = None
                metadata["pd_l1_available"] = False

        elif "tmb" in l:
            match = re.search(r"(\d+)", lab)
            if match:
                metadata["tmb"] = int(match.group(1))
                metadata["tmb_available"] = True
            else:
                metadata["tmb"] = None
                metadata["tmb_available"] = False

        elif "msi-h" in l:
            metadata["msi_h"] = True
            metadata["msi_h_available"] = True

        elif "progressive" in l:
            metadata["progressive_disease"] = True
            metadata["progressive_disease_available"] = True

        elif "prior therapy failure" in l:
            metadata["prior_therapy_failure"] = True
            metadata["prior_therapy_failure_available"] = True

        elif "+" in lab:
            if isinstance(metadata.get("mutations"), list):
                metadata["mutations"].append(lab.strip())
            else:
                metadata["mutations"] = [lab.strip()]
            metadata["mutations_available"] = True

    return metadata


def transform_patient_data():
    df_cases = pd.read_csv("synthetic_cases.csv")
    processed_data = []
    for idx, row in df_cases.iterrows():
        idx = int(idx)
        labs = eval(row["labs"]) if isinstance(row["labs"], str) else row["labs"]
        metadata = parse_labs_with_flags(labs)

        processed_data.append(
            {
                "id": f"case_{idx + 1:04d}",
                "document": row["case_text"],
                **metadata,
                "risk_score": row["risk_score"],
                "decision": row["decision"],
            }
        )

    df_processed = pd.DataFrame(processed_data)
    output_file = "synthetic_cases_processed.csv"
    df_processed.to_csv(output_file, index=False)


def create_or_get_chroma_client():
    # Use OpenAI embeddings
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db_cases"
    )  # persistent storage
    collection_name = "synthetic_cases"
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
    )
    if collection_name in [c.name for c in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(collection_name)
    else:
        collection = chroma_client.create_collection(
            name=collection_name, embedding_function=openai_ef
        )
    return collection


def generate_embedding():
    ids = []
    documents = []
    metadatas = []
    collection = create_or_get_chroma_client()
    df_synthetic_cases_processed = pd.read_csv("synthetic_cases_processed.csv")
    for _, row in df_synthetic_cases_processed.iterrows():
        ids.append(row["id"])
        documents.append(row["document"])

        metadata = {
            "pd_l1": float(row["pd_l1"]),
            "pd_l1_available": bool(row["pd_l1_available"]),
            "tmb": float(row["tmb"]),
            "tmb_available": bool(row["tmb_available"]),
            "msi_h": bool(row["msi_h"]),
            "msi_h_available": bool(row["msi_h_available"]),
            "progressive_disease": bool(row["progressive_disease"]),
            "progressive_disease_available": bool(row["progressive_disease_available"]),
            "prior_therapy_failure": bool(row["prior_therapy_failure"]),
            "prior_therapy_failure_available": bool(
                row["prior_therapy_failure_available"]
            ),
            "mutations": (
                row["mutations"]
                if isinstance(row["mutations"], str)
                else str(row["mutations"])
            ),
            "mutations_available": bool(row["mutations_available"]),
            "risk_score": float(row["risk_score"]),
            "decision": row["decision"],
        }

        metadatas.append(metadata)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        count = collection.count()
        logger.info(f"Total cases in collection: {count}")


def fetch_guidelines_db():
    with open("guidelines_db.pkl", "rb") as _guideline:
        guideline_instructions = pickle.load(_guideline)
    return guideline_instructions

def get_proper_user_input():
    user_option_list = ["approve", "reject", "require_additional_information",
                        ]
    message = f"Review provided options and choose any one of them. {user_option_list}\nSelect your input:"
    # logger.info(message)
    user_input = input(message)

    if user_input not in user_option_list:
        logger.warning(f"Selected options is not right category. Please choose another.")
        get_proper_user_input()
    else:
        return user_input

@tool
def get_patient_data_by_id(id: str) -> Dict:
    """This method used to fetch patient details based upon given patient_id.
    :args id: Input patient id in str format.
    :return: patient_data: For correct patient id mapped part, it will return patient object else None.
    """
    patient_data = get_patient_data()
    for _patient_obj in patient_data:
        if id == _patient_obj["patient_id"]:
            return _patient_obj


@tool
def perform_similarity_search(
    patient_query: str, max_results: int = TOP_K
) -> Dict[str, Any]:
    """
    This method is used to perform similarity search from chromadb for given patient_query
    :param patient_query: patient query in str format.
    :param max_results: Based upon constant value, system will pull topx K results from chromaDB.
    :return: Dict object containing fetched similar searched patient details else None
    """
    similar_patient_case_details: Optional[Dict]
    collection = create_or_get_chroma_client()
    results = collection.query(
        query_texts=[patient_query],
        n_results=max_results,
        # where={"$and": filter_condition},
    )
    if results:
        similar_patient_case_details = {
            "ids": results["ids"][0],
            "documents": results["documents"],
            "metadata": results["metadatas"][0],
            "distances": results["distances"][0],
        }
    return similar_patient_case_details


@tool
def fetch_guideline_api(condition: str) -> str:
    """
    This method is used to provide ongoing guideline evidences for similar conditions in real world.
    :param condition: user provided topic condition in str
    :return str based text content else None:
    """

    params = {"query.term": condition, "pageSize": 1}

    response = requests.get(GUIDELINE_API_URL, params=params)

    if response.status_code != 200:
        return f"Error fetching guidelines: {response.status_code}"

    data = response.json()

    if "studies" in data and len(data["studies"]) > 0:
        study = data["studies"][0]
        desc = (
            study.get("protocolSection", {})
            .get("descriptionModule", {})
            .get("briefSummary", "No summary available")
        )
        return desc

    return "No current trial data found."


@tool
def search_pubmed_api(query: str, max_results=3) -> List:
    """
    This method is used to provide recent advancement and research happening in the oncology field using the pubmed api.
    :param query: user provided topic of research.
    :param max_results: total providing max results
    :return: list containing abstract results.
    """

    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results}

    response = requests.get(PUBMED_API_URL, params=params)

    # Uses NCBI E-utilities esearch endpoint.
    # Searches the PubMed database (db=pubmed).
    # Requests results in JSON format (retmode=json).
    # Limits the number of retrieved IDs to max_results.
    # Output: A JSON response containing a list of PubMed article IDs (PMIDs) relevant to the search term.
    ids = response.json().get("esearchresult", {}).get("idlist", [])

    # Prepares a list to store the text of each abstract.
    abstracts = []

    # Loop Over IDs and Fetch Abstracts
    for pmid in ids:

        fetch_params = {
            "db": "pubmed",  # target database
            "id": pmid,  # specific article
            "retmode": "text",  # return plain text
            "rettype": "abstract",  # return only the abstract
        }

        abs_resp = requests.get(PUBMED_API_URL, params=fetch_params)

        abstracts.append(abs_resp.text.strip())
    return abstracts


@tool
def calculate_risk_api(json_labs: str) -> float:
    """
    This method is used to calculates risk score for advanced treatment recommendation
    using a lookup table of biomarker weights. Base risk = 0.3, final score capped at 1.0
    :param json_labs: json object containing string information for labs containing results.
    :return float based value for risk score.
    """
    score = 0.3
    labs = json.loads(json_labs)

    for l in labs:
        text = l.lower()

        # Handle PD-L1 explicitly (numeric value)
        if "pd-l1" in text and "%" in text:
            try:
                pd = int(l.split(":")[1].replace("%", "").strip())
                score += min(pd * BIOMARKER_WEIGHTS["pd-l1"], 0.4)
            except Exception:
                pass

        # Handle TMB explicitly (numeric value)
        elif "tmb" in text:
            try:
                tmb = float(l.split(":")[1].replace("mut/mb", "").strip())
                score += min(tmb * BIOMARKER_WEIGHTS["tmb"], 0.1)
            except Exception:
                pass

        # Handle other predefined markers
        else:
            for marker, weight in BIOMARKER_WEIGHTS.items():
                if marker in text:
                    score += weight

    return round(min(score, 1.0), 2)


@tool
def process_claim_details_and_decide_claim_status(json_context: str) -> Dict[str, Any]:
    """This method is used to decide the claim status based upon the provided prompt details.
    :param json_context: Context as a json object for providing dynamic details for patient.
    :return a dict object containing claim result.
    """
    parser = JsonOutputParser(pydantic_object=LLMResponseSchema)
    insurance_prompt = generate_prompt_with_similar_search_content_and_risk_score()
    context = json.loads(json_context)
    prompt = ChatPromptTemplate.from_template(
        insurance_prompt,
        partial_variables={
            "risk_score": context["risk_score"],
            "similarity_search_result": context["similarity_search_result"],
        },
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm | parser
    response_dict = chain.invoke(
        {"patient_medical_history": context["patient_medical_history"]}
    )
    response_dict = {"status": "approve", "summary": "this is a mock response."}
    logger.debug(f"Response dict: {response_dict}")
    return response_dict


@tool
def provide_additional_claim_response_using_pubmed_and_guideline_api(
    json_context: str,
) -> Dict[str, Any]:
    """This method is used to provide recent advancements happening in the oncology using pubmed and guideline api.
    :param json_context: Context as a json object for providing dynamic details for patient.
    :return a dict object containing claim result.
    """
    parser = JsonOutputParser(pydantic_object=LLMResponseSchema)
    insurance_prompt = generate_prompt_with_all_possible_parameters()
    context = json.loads(json_context)
    prompt = ChatPromptTemplate.from_template(
        insurance_prompt,
        partial_variables={
            "risk_score": context["risk_score"],
            "pubmed_api_result": context["pubmed_api_result"],
            "guideline_api_result": context["guideline_api_result"],
            "similarity_search_result": context["similarity_search_result"],
        },
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = prompt | llm | parser
    response_dict = chain.invoke(
        {"patient_medical_history": context["patient_medical_history"]}
    )
    response_dict = {"status": "approve", "summary": "this is a mock response."}
    logger.debug(f"Response dict: {response_dict}")
    return response_dict


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
        "require_additional_information",
        "approve",
        "reject"
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
        # claim_status_obj = process_claim_details_and_decide_claim_status.invoke(
        #     json.dumps(dynamic_patient_dict)
        # )
        obj_1 = {"summary": "Mock response from reasoning agent. Currently getting error with openai api key.",
                            "claim_status":"approve"}
        obj_2 = {"summary": "Mock response from reasoning agent. Currently getting error with openai api key.",
                            "claim_status":"human review"}
        # json_obj_list = [json.dumps(obj_1), json.dumps(obj_2)]
        # json_obj_list = [json.dumps(obj_1)]
        # json_obj_list = [obj_1]
        json_obj_list = [obj_2]
        # json_obj_list = [obj_1, obj_2]
        claim_status_dict = random.choice(json_obj_list)
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
        pubmed_search_content = json.dumps([f"I'm a mock content of pubmed. Error is coming from the pubmed api."])
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
        # claim_status_obj = (
        #     provide_additional_claim_response_using_pubmed_and_guideline_api.invoke(
        #         json.dumps(dynamic_patient_dict)
        #     )
        # )
        # claim_status_dict = json.loads(claim_status_obj)
        obj_1 = {"detailed_summary": "Mock response from additional detail provider agent. "
                            "Currently getting error with openai api key."}
        # json_obj_list = [json.dumps(obj_1), json.dumps(obj_2)]
        # json_obj_list = [json.dumps(obj_1)]
        json_obj_list = [obj_1]
        claim_status_dict = random.choice(json_obj_list)
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
    if llm_suggested_claim_status == "reject" and (not human_feedback):
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
    elif llm_suggested_claim_status == "approve"  and (not human_feedback):
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
            "conclude_claim_agent": "conclude_claim_agent"
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
    graph_app = graph_builder.compile(checkpointer=memory_saver, interrupt_after=["involve_human_review_agent"])
    logger.info("end: compile_graph")
    return graph_app, memory_saver


def process_agentic_solution():
    logger.info("start: main function.")
    try:
        app, memory = compile_graph()
        png_data = app.get_graph().draw_mermaid_png()
        file_path = os.path.join(os.getcwd(), "workflow.png")
        with open(file_path, "wb") as f:
            f.write(png_data)
        print(f"filed saved at: {file_path}")
        user_id = "P001"
        config = {"configurable": {"thread_id": user_id}}
        user_query = f"Please process claim for patient_id: {user_id}"
        initial_state = {
                "patient_id": user_id,
                "current_message": user_query,
                "conversation_message": [HumanMessage(user_query)],
                "human_review_feedback": "not_initiated"
            }
        final_resp = None
        for event in app.stream(initial_state, config, stream_mode="values"):
            final_resp = event
        while final_resp["human_review_feedback"] not in ("approve", "reject"):
            user_input = get_proper_user_input()
            app.update_state(config, {"human_review_feedback": user_input},
                             as_node="involve_human_review_agent")
            for event in app.stream(None, config, stream_mode="values"):
                final_resp = event
        logger.info("end: main function.")
        return final_resp
    except Exception as e:
        logger.info("end: main function.")
        logger.exception("Error while processing request: {e}")

final_resp = process_agentic_solution()
logger.info(f"FINAL RESULT: {final_resp['final_message']}")