# Deployment
There are two ways the web-app can be deployed on the host machine. Either docker container can be deployed or web-app can be started after building project from the source.

### 1. Deploy Docker container for web-app

If docker is not installed follow [this](https://docs.docker.com/engine/install/) link.

1. Execute startup script.

```bash start.sh```

The script first checks if docker image is locally available. If not, image is 
pulled from the docker hub.

If any error is encountered during docker deployment, follow the following steps:

1. Fetch docker image for the project: ```docker pull t6nand/group07-co2```

2. Deploy container: `docker run -d -p 8050:8050 t6nand/group07-co2`

### 2. Build from the source:

1: Setup virtual environment.

```python3 -m venv ./venv```  
```source venv/bin/activate```

2. Install dependencies

```pip install -r requirements.txt```

3. Start the web-app

```python src/app.py```

Server is then available at http://127.0.0.1:8050


# Web Application Requirement

Requires Python 3.5+

## Start server: 
```python src/app.py``` 

## Open web app:
```http://127.0.0.1:8050/```
