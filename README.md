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