import logging
from logging_config import setup_logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Literal
from langchain_core.messages import HumanMessage
from execute_graph import compile_graph


class UserRequest(BaseModel):
    patient_id: str


class UpdateClaimRequest(BaseModel):
    patient_id: str
    human_review_feedback: Literal[
        "require_additional_information", "approve", "reject"
    ]


setup_logging()
_logger = logging.getLogger("api")

app = FastAPI()
graph_app = None
graph_memory = None


@app.on_event("startup")
async def startup_event():
    global graph_app, graph_memory
    graph_app, graph_memory = compile_graph()
    _logger.info("Graph compiled and ready to use.")


@app.post("/process-claim")
async def process_claim(user_request: UserRequest, request: Request):
    try:
        global graph_app, graph_memory
        patient_id = user_request.patient_id
        config = {"configurable": {"thread_id": patient_id}}
        if not graph_app:
            raise AttributeError()
            # Start fresh
        print(f"Starting new session for patient: {patient_id}")
        user_query = f"Please process claim for patient_id: {patient_id}"
        initial_state = {
            "patient_id": patient_id,
            "current_message": user_query,
            "conversation_message": [HumanMessage(user_query)],
            "human_review_feedback": "not_initiated",
        }
        result = graph_app.invoke(initial_state, config=config)
        _logger.info(f"Result from fresh claim execution: {result}")
        return result

    except AttributeError:
        raise HTTPException(detail="error: Graph not initialized", status_code=500)
    except Exception:
        raise HTTPException(detail="Something went wrong.", status_code=500)
    finally:
        pass


@app.post("/process-claim-with-human-feedback")
async def process_claim_with_human_feedback(update_claim:UpdateClaimRequest, request: Request):
    try:
        global graph_app, graph_memory
        patient_id = update_claim.patient_id
        human_review_feedback = update_claim.human_review_feedback
        config = {"configurable": {"thread_id": patient_id}}
        stored_state = graph_memory.get(config)
        if stored_state and "channel_values" in stored_state:
            # Continue from existing memory
            print(f"Resuming for Patient: {patient_id}")
            channel_values = stored_state["channel_values"]
            current_message = channel_values["current_message"]
            message_history = channel_values["conversation_message"]
            if not message_history:
                message_history = [f"{HumanMessage(human_review_feedback)}"]
            else:
                message_history.append(f"{HumanMessage(human_review_feedback)}")
            updated_state = {
                "current_message": current_message,
                "conversation_message": message_history,
                "patient_id": patient_id,
                "human_review_feedback": human_review_feedback
            }
            graph_app.update_state(
                config,
                updated_state
            )
            result = graph_app.invoke(None, config=config)
            print(f"result from memory+update: {result}")
            return result
        else:
            raise HTTPException(detail="Patient details not found. Please check again.", status_code=404)
    except AttributeError:
        raise HTTPException(detail="error: Graph not initialized", status_code=500)
    except Exception:
        raise HTTPException(detail="Something went wrong.", status_code=500)
    finally:
        pass