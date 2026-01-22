# Installation Instructions
It is assumed that docker has been installed and configured with the NVIDIA container toolkit. Also ensure that the host system's NVIDIA driver version is compatible with the docker container image being used. 

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
Open `1_cuvs-bench_data_preparation.ipynb` in JupyterLab. Modify parameters such as `ROOT`, `RAW_DATA_PATH`, `DATASET_NAME` and `NUM_SAMPLES` in the first executable cell. The input data is a collection of nested numpy files. By default, the sampled data output will be in the `datasets` directory. 

## Running KMeans Clustering
Several GPU accelerated KMeans clustering algorithms are provided in the NVIDIA RAPIDS docker container. The following docker command will launch a new JupyterLab instance within a docker container with GPU support after installing a few `conda` packages. 

```
IMAGE_NAME="rapidsai/base:25.10-cuda12.0-py3.11"
HOST_DIR=$(pwd)  # Current directory
CONTAINER_DIR="/myworkspace"
docker run -it --rm \
    --net=host \
    --gpus all \
    -v "${HOST_DIR}:${CONTAINER_DIR}" \
    "${IMAGE_NAME}" bash -c "

    cd /myworkspace
    conda install -y -c conda-forge -c pytorch -c nvidia faiss-gpu jupyterlab
    apt update
    apt install htop nvtop --assume-yes

    jupyter lab
    "
```

Open the `kmeans_clustering.ipynb` notebook and modify the parameter values as needed. By default, datasets are stored in `./datasets`. Dataset names are provided as a list. The notebook will sweep across `algorithms`, `dataset_names`, and `k_values`. Intermediate results are stored in the `./results` folder. 

Note that certain errors can cause the sweep to not advance (e.g., OOM issues). You can manually stop the notebook by pressing the square button on the JupyterLab UI. It should continue with the next run in the test grid. 

Once all runs have been completed, the results will be merged into `merged_kmeans_results.csv` in the root folder.

### Clean Up
Docker will continue running in the JupyterLab instance in the background. Use `docker ps` to list all active Docker containers:
```
CONTAINER ID   IMAGE                                 COMMAND                  CREATED        STATUS        PORTS     NAMES
a68263c957b6   rapidsai/base:25.10-cuda12.0-py3.11   "/home/rapids/entryp…"   23 hours ago   Up 23 hours             wizardly_hellman
```

Then stop the container by running `docker stop <CONTAINER_ID>`:
```
docker stop a68263c957b6
```