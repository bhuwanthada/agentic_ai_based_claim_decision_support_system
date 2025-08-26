import os
from typing import Dict
from langchain_core.messages import HumanMessage, SystemMessage
import logging
from logging_config import setup_logging
from lg_workflow import compile_graph
from utils import get_proper_user_input

setup_logging()
logger = logging.getLogger("execute_graph")


def process_agentic_solution(patient_id: str) -> Dict:
    """
    Method to invoke agentic ai solution.
    :param patient_id:
    :return:
    """
    logger.info("start: main function.")
    try:
        app, memory = compile_graph()
        png_data = app.get_graph().draw_mermaid_png()
        file_path = os.path.join(os.getcwd(), "workflow.png")
        with open(file_path, "wb") as f:
            f.write(png_data)
        logger.info(f"filed saved at: {file_path}")
        user_id = patient_id
        config = {"configurable": {"thread_id": user_id}}
        response = memory.get(config)
        user_query = f"Please process claim for patient_id: {user_id}"
        initial_state = {
            "patient_id": user_id,
            "current_message": user_query,
            "conversation_message": [HumanMessage(user_query)],
            "human_review_feedback": "not_initiated",
        }
        final_resp = None
        for event in app.stream(initial_state, config, stream_mode="values"):
            final_resp = event
        logger.info("*"*20)
        logger.info(f"Message: patient id: {user_id} "
                    f"medical examination shows: {final_resp.get('patient_medical_details')} "
                    f" and agentic system suggests feedback as: {final_resp.get('current_message')}")
        while final_resp["human_review_feedback"] not in ("approve", "reject"):
            user_input = get_proper_user_input()
            app.update_state(
                config,
                {"human_review_feedback": user_input},
                as_node="involve_human_review_agent",
            )
            for event in app.stream(None, config, stream_mode="values"):
                final_resp = event
        logger.info("end: main function.")
        return final_resp
    except Exception as e:
        logger.info("end: main function.")
        logger.exception("Error while processing request: {e}")



