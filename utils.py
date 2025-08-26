import pickle
import re
import pandas as pd
from chromadb.utils import embedding_functions
import chromadb
import os

from pydantic import BaseModel, Field
from typing import Dict, List, Literal

from dotenv import load_dotenv
import logging
from logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger("utils")


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
    user_option_list = [
        "approve",
        "reject",
        "require_additional_information",
    ]
    message = f"Review provided options and choose any one of them. {user_option_list}\nSelect your input:"
    # logger.info(message)
    user_input = input(message)

    if user_input not in user_option_list:
        logger.warning(
            f"Selected options is not right category. Please choose another."
        )
        get_proper_user_input()
    else:
        return user_input
