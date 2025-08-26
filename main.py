import logging
from logging_config import setup_logging
from execute_graph import process_agentic_solution
setup_logging()
logger = logging.getLogger("main")

if __name__ == "__main__":
    try:
        user_input_patient_id = input("Please provide patient id to process.\nEnter patient id:")
        final_resp = process_agentic_solution(user_input_patient_id)
        # logger.info(f"final result: {final_resp}")
        logger.debug("*"*20)
        logger.debug(f"Message: patient id: {user_input_patient_id} "
                    f"medical examination shows: {final_resp.get('patient_medical_details')} "
                    f" and agentic system suggests feedback as: {final_resp.get('llm_suggested_claim_message')}."
                    f"The Final status including human review shows the claim status as: {final_resp['final_message']}")

        logger.info("*"*25)
        logger.info("*"*25)
        logger.info("*"*25)
        logger.info("*"*25)
        logger.info("*"*25)
        logger.info(f"Detailed claim details: {final_resp}")
    except Exception:
        logger.exception("Error while processing request. Please try again")
    finally:
        logger.info("Process ended.")
