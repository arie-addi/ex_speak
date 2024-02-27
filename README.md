# Booking portal exercise
# by: Arie Addi
# 26-Feb-2023
# NLP Exercise for Speak Now
 
This application analyzes English audio or text samples of an individual and gives an overall rating on a number of parameters or rubrics such as vocabulary and grammar. There is a command line interface as well as an API. For ease of use a very basic Web interface was also implemented demonstrating the central API requests.

## Goal
A major focus of my coding focused on establishing a python class that would be extensible and easy to test with multiple models. In data analysis a good deal of trial and error is commonly done to fine tune a model for best results. That is why the class includes baseline results to compare to on training. It is common in ML to find a better model to adopt in future releases, so I class that can facilitate this is quite useful. See more details in report file.

## Requirements

 - Docker https://docs.docker.com/engine/installation/
 - Docker Compose https://docs.docker.com/compose/install/
 - Initial files in directories data/ assets/ and static/
 - Training must be performed before running model unless `joblib` file exists for specific model

Note: A tar file can be downloaded to create most of the default data files

## Usage

To boot up the application to run, docker compose is used:

```
$ docker compose build
$ docker compose up
```

To facilitate the setup of docker such as the commands above and the needed default data, a shell script is included:

```
$ bash ./run_docker.sh
```

A list of common commands to manage this docker build is displayed at completion of this script. This includes how to attach to the app shell console as well as the FastAPI console.

The application will be available at http://localhost:8000 If for any reason you need the port to be different to 8000, you can change the port mapping in docker-compose.yml. For example to expose the application on port 9292 change `- "8000:8000"` to `- "9292:8000"`.

### Web interface

Open on your browser

```
http://localhost:8000/index.html
```

### API

To see the API interface options and parameters open your browser to:

```
http://127.0.0.1:8000/openapi.json
```

Example Requests

```
$ curl http://localhost:8000/api; echo
{"message":"Hello World"}

# default analysis is 'vocab_avg'
$ curl http://localhost:8000/transcriptions/1647885061811312/analyze?category=text; echo
{"id":1647885061811312,"category":"text","vocab_avg":4.33}

# use a new csv file 'input_smA.csv' in data directory with the same 5 transcript columns with assessment_id as index
$ curl http://localhost:8000/transcriptions/1667330281849999/analyze?category=text&test_data=input_smA.csv; echo
{"id":1667330281849999,"category":"text","vocab_avg":5.0}
```

Note: API support is only for a single assessment. Would be trivial to return a JSON array of assessments as well. The CLI
does this by writing the results to a csv file instead.

### Command Line Interface

The docker shell is needed to run the CLI, `main_cli.py`. It can be reached after 
docker is started by running the command:

```
$ docker attach main-bash-1
root@0f3e6f7020b2:/app# ./main_cli.py --help
```

Note: Command line arguments are managed by the python library 'click' and is quite slow to startup.
There are two sub-commands to run the CLI: 'run-training' and 'analyze'
There are many options which are well defined with the --help argument

```
$ ./main_cli.py run-training --help
$ ./main_cli.py analyze --help
```

Example Commands

```
$ ./main_cli.py run-training --category text --target vocab_avg 
$ ./main_cli.py analyze --model CATBOOST --log_level warning --test_data input_smA.csv
$ ./main_cli.py analyze --log_level error --test_data input_smA.csv --asset 1692530001326214
```

Unless the --asset flag is used results of the analsis are saved in column `evaluate_predicted` in the csv file:
   `data/run_results_<target_name>.csv`

Note: The default values in the command line arguments are controlled by:
   - The modelD dictionary in file `model_util.py`
   - @click.option() lines in file `main_cli.py`
   - in __init__ constructor method in class `ModelLanguageQuality`

Overview of Files Used

`model_language_quality.py` - class where all the functionality for training and running models can be found
`model_util.py` - basic data structures for configuration as well as generic functions defined
`main_cli.py` - CLI for initializing class `ModelLanguageQuality`
`main.py` - REST API for FastAPI initializing as well class `ModelLanguageQuality`
`collect_transcripts.py` - Util to create static index page based on transcript files that exist in directories `/audio` and `/text`
