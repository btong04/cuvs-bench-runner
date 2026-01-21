# Installation Instructions
Navigate to your fast storage drive (e.g., `/raid` on local machine). The scripts will sample and convert data under it's root directory by default. 

Clone git repository:
```
git clone https://github.com/btong04/cuvs-bench-runner.git
cd /raid/cuvs-bench-runner
```

Create a pip virtual environment and install required packages.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For subsequent runs, you can navigate to the script directory and run `source venv/bin/activate`. Launch JupyterLab to start an interactive notebook sessions: 
```
jupyter lab
```

Click on the JupyterLab link (e.g., `http://localhost:8888/lab?token=xxx`) provided in the output of the command above. If you are working on a remote machine, make sure to set up forwarding on port 8888 when connecting to the remote instance (`ssh nvidia@remote-machine-name -L 8888:localhost:8888`).

# Getting Started
Open `1_cuvs-bench_data_preparation.ipynb` in JupyterLab. Modify parameters such as `ROOT`, `RAW_DATA_PATH`, `DATASET_NAME` and `NUM_SAMPLES` in the first executable cell. The input data is a collection of nested numpy files. 


## Running KMeans Clustering
Several GPU accelerated KMeans clustering algorithms are provided in the NVIDIA RAPIDS docker container. The following command will launch a new JupyterLab instance within a docker container. 

```
IMAGE_NAME="rapidsai/base:25.10-cuda12.0-py3.11"
HOST_DIR=$(pwd)  # Current directory
CONTAINER_DIR="/myworkspace"
docker run -it --rm \
    --net=host \
    --gpus all \
    -v "${HOST_DIR}:${CONTAINER_DIR}" \
    -v "/raid/embeddings:/embeddings" \
    "${IMAGE_NAME}" bash -c "

    cd /myworkspace
    conda install -y -c conda-forge -c pytorch -c nvidia faiss-gpu jupyterlab
    # pip install -r requirements.txt
    apt update
    apt install htop nvtop --assume-yes

    jupyter lab
    "
```