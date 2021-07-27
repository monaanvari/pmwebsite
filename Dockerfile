FROM python:3.8-slim 
COPY . ./
CMD streamlit run main.py --server.port $PORT