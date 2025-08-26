import json
import requests
from langchain_core.tools import tool
from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from constant import (
    BIOMARKER_WEIGHTS,
    TOP_K,
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
from utils import get_patient_data, create_or_get_chroma_client
from models import LLMResponseSchema, LLMResponseSchemaForDetailedSummery

load_dotenv()
setup_logging()
logger = logging.getLogger("langchain_tools")


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

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm | parser
    response_dict = chain.invoke(
        {"patient_medical_history": context["patient_medical_history"]}
    )
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
    parser = JsonOutputParser(pydantic_object=LLMResponseSchemaForDetailedSummery)
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

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm | parser
    response_dict = chain.invoke(
        {"patient_medical_history": context["patient_medical_history"]}
    )
    logger.debug(f"Response dict: {response_dict}")
    return response_dict
