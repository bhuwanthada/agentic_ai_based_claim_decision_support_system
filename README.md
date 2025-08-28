# Purpose
This Notebook is mainly used to review the oncology departments claims in the Insurance domain.

# Technical Usage
1. Agentic rag side, used the langchain and langgraph.
2. For API endpoint, used Fastapi
3. For UI representation, used streamlit
4. For Database, few .pkl file and chromadb used for vector emnedding.
5. For providing historical data, used a provided csv file.

# Steps to execute
Make sure you have python 3.10+ version available.
Provide OPENAI API KEY in the .env file.
In the terminal Run command: ```python -m venv <venv_name>``` to create the virtual environment.
In the terminal Run command: ```pip install -r requirements.txt``` to install all necessary packages.
In the terminal Run command: ```python main.py``` and provide user input on cmd.
Observe the system responses over there and provide necessary responses wherever require.
Upon providing correct response, application will show you necessary result.
You can check application logs in logs folder containing log file mentioned with each date.

# Steps to execute using REST API as backedn and Streamlit as UI
Make sure you have python 3.10+ version available.
Provide OPENAI API KEY in the .env file.
In the terminal Run command: ```python -m venv <venv_name>``` to create the virtual environment.
In the terminal Run command: ```pip install -r requirements.txt``` to install all necessary packages.
In the terminal Run command: ```uvicorn manage:app --realod``` . This will make app up and running on localhost:8000/docs
In the another terminal Run command: ```streamlit run ui.py``` This will make UI up and running on localhost:8501

You will see a list of patient details in unprocessed section part. Click on different tabs present over there. There are few api's calls will going to happen to pull claim details and taking human review for final decision.
