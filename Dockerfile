FROM python:3.7.9
LABEL maintainer="tapan.sharma@tum.de"
WORKDIR /group07
ADD requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8050
CMD ["python", "./src/app.py" ]