# **Gold Price prediction**

## Project Description

This project aims to develop a machine learning model to predict the future gold price. Gold is a precious metal with a long history of being used as a currency and store of value. However, its price is volatile and can be difficult to predict. A machine learning model could be used to identify patterns in historical data and use those patterns to make predictions about future prices


## Data
The data for this project was obtained from the World Gold Council. It consists of daily gold prices from January 2008 to June 2018. The data includes the following features:



# **Project Setup**

## Virtual Environment and Dependencies

To run the Jupyter notebooks in this project or a web service, is required to set up a virtual environment and install the dependencies.

1. Clone the repository to your local machine and navigate to the project directory (root):

    ```sh
    git clone https://github.com/penscola/Gold-Prediction.git
    ```

2. Ensure that `python` is installed. If it is not installed, use the following command:
    
        ```sh
        sudo apt-get install python3.8

3. Install the `requirment.txt` using the command:
    
        ```sh
        pip3 install -r requirements.txt
        ```

4. To start the Jupyter notebook run:

    ```sh
    jupyter notebook
    ```
    
And it will be started the Jupyter Notebook in the virtual environment context on the browser. Just go to the [`notebooks/notebook.ipynb`](https://github.com/penscola/Gold-Prediction/blob/master/src/notebook/notebook.ipynb) and try to run the notebook.

# **Web Service Deployment**

## Virtual Environment Setup

To host the service locally using the virtual environment, run the python script [`src/predict.py`](https://github.com/penscola/Gold-Prediction/blob/master/src/predict.py)  to start the Flask application:

```sh
python3 src/predict.py
```

With the Flask application running, we can make HTTP requests to port 9696. For example, in the Jupyter notebook located in [`notebooks/notebook.ipynb`](https://github.com/penscola/Gold-Prediction/blob/master/src/notebook/notebook.ipynb):

```python
url_local = "http://127.0.0.1:9696/predict"
test_data = {
    'SPX' : 3000.12,
    'USO' : 11.0,
    'SLV' : 15.1,
    'EUR/USD' : 1.1,
}
test_data_values = list(test_data.values())
requests.post(url_local, json = test_data_values).json()
```

## Docker Deployment

1. Open a terminal or command prompt. Navigate to the directory containing the Dockerfile. Run the following command to build the Docker image named diabetes_prediction (you can give a different name to the image if you prefer):

        sudo docker build -t diabetes_prediction .

        or

        sudo docker pull marcosbenicio/diabetes_prediction:latest

2. To list all the Docker images on your system and verify that the image is there, use:

        sudo docker images

3. After the image is built or pushed, run a container from it with the following command:

        sudo docker run -p 9696:9696 prediction

    With the Flask application running inside Docker, we can make HTTP requests to port 9696. For example, in the Jupyter notebook located in [`notebooks/notebook.ipynb`](https://github.com/penscola/Gold-Prediction/blob/master/src/notebook/notebook.ipynb), run the following code cell at the section 5.2: 

    ```python
            url_local = "http://127.0.0.1:9696/predict"
        test_data = {
            'SPX' : 3000.12,
            'USO' : 11.0,
            'SLV' : 15.1,
            'EUR/USD' : 1.1,
        }
        test_data_values = list(test_data.values())
        requests.post(url_local, json = test_data_values).json()
    ```

### Cloud Deployment

The model was deployed to the cloud using HuggingFace Spaces.

1. Create an account on [HuggingFace](https://huggingface.co/) Spaces.
2. Create a new space and upload the model file [`model.pkl`](model/Random-Forest-Regressor.pkl) and the file [`app.py`](src/app.py) to the space.
3. Deploy the model to the cloud using the HuggingFace Spaces interface.

