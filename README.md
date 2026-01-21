# Installation Instructions
Navigate to fast storage disk (e.g., `/raid` on local machine). Clone git repository:
```
git clone https://github.com/btong04/cuvs-bench-runner.git
cd ./cuvs-bench-runner
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
Open `1_cuvs-bench_data_preparation.ipynb` in JupyterLab. Modify parameters such as `ROOT`, `RAW_DATA_PATH`, `DATASET_NAME` and `NUM_SAMPLES` in the first executable cell. The input data is a collection of numpy files.
