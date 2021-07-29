#Base Image to use
FROM python:3.8-slim

#Expose port 8080
EXPOSE 8080

#Copy Requirements.txt file into app directory
COPY requirements.txt pmwebsite/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r pmwebsite/requirements.txt

#Copy all files in current directory into app directory
COPY . /pmwebsite

#Change Working Directory to app directory
WORKDIR /pmwebsite

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "program.py", "--server.port=8080", "--server.address=0.0.0.0"]
~                                                                                                       