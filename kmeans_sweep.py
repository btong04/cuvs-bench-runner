import os
import sys
import re
import glob
import logging
import numpy as np
import struct
import pandas as pd
import threading
import subprocess
import time
from tqdm import tqdm
from functools import wraps
import argparse

parser = argparse.ArgumentParser(description="Script for benchmarking KMeans.")

parser.add_argument('-dataset_name', type=str, default='miracl-fp32-1024d-1M', help="Dataset name.")
parser.add_argument('-dataset_path', type=str, default='./datasets', help="Dataset path.")
parser.add_argument('-algorithm', type=str, default='cuvs_kmeans_balanced', help="Select KMeans clustering algorithm from: 'cuvs_kmeans_balanced', 'cuvs_kmeans', 'faiss_cpu_kmeans', or 'faiss_gpu_kmeans'. Default is 'cuvs_kmeans_balanced'.")
parser.add_argument('-apply_scaler', type=bool, default=False, help="Apply StandardScaler to rescale input vectors. Default is False.")
parser.add_argument('-max_iter', type=int, default=20, help="Maximum number of iterations for KMeans training. Default is max_iter=20.")
parser.add_argument('-k_values', type=str, default='[10,100,1000]', help="List of k-values (i.e., number of clusters) represented as a string. Default is k_values='[10,100,1000]'.")

# Parse input arguments
args = parser.parse_args()
DATASET_NAME = args.dataset_name
DATASET_PATH = args.dataset_path
algorithm = args.algorithm
apply_scaler = args.apply_scaler
max_iter = args.max_iter
k_values = args.k_values

print(f'Running {algorithm}. \n')

# Parse k_value str into list of int
k_values = list(map(int, re.findall(r"\d+", k_values)))

# Path to base.fbin files in dataset
base_path = f'{DATASET_PATH}/{DATASET_NAME}/base.fbin'

# Validate algorithm selection
if algorithm in ['faiss_cpu_kmeans']:
    hw_type = 'cpu'
elif algorithm in ['faiss_gpu_kmeans', 'cuvs_kmeans', 'cuvs_kmeans_balanced']:
    hw_type = 'gpu'
    
    if algorithm == 'cuvs_kmeans_balanced':
        cuvs_kmeans_extra_params = {'hierarchical': True, 'hierarchical_n_iters': max_iter}
    else:
        cuvs_kmeans_extra_params = {}
else:
    raise ValueError(f"Invalid hw_type={hw_type} provided. Please choose 'cpu' or 'gpu'.")


def add_prefix(d, prefix):
    """
    Add prefix to dictionary keys.
    """
    return {f"{prefix}{k}": v for k, v in d.items()}
    
def monitor_resources(stop_event, log, hw_type, interval=0.25):
    """
    Monitor CPU and RAM usage until stop_event is set.

    interval: float
      Polling time in seconds.
    """
    import psutil
    import time
    
    if hw_type == 'gpu':
        import pynvml
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Get device handle for GPU_0

    while not stop_event.is_set():
        cpu_util = psutil.cpu_percent(interval=interval)  # Measures over 1 second
        ram_util = psutil.virtual_memory().used/(1024**3)

        sys_util = {'timestamp': time.perf_counter(), 'cpu_util': cpu_util, 'ram_gb': ram_util}

        if hw_type == 'gpu':
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_mem_used_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).used/(1024**3)
            gpu_util = {'gpu_util': gpu_utilization, 'vram_gb': gpu_mem_used_gb}

            # Append GPU telemetry to system telemetry
            sys_util = sys_util | gpu_util

        # print(sys_util)
        
        # Store system logs
        log.append(sys_util)
    
    if hw_type == 'gpu':
        pynvml.nvmlShutdown()
    
    return(log)

def summarize_telemetry(resource_log, hw_type):
    """
    Summarize telemetry data.

    resource_log: list of dict
      Resource monitoring logs stored as a list of dictionary.
    stage_prefix: str
      Prefix to add to output dictionary. Choose something like 'idx_build', 'vec_search'.
    hw_type: str
      Hardware used. Select 'cpu' or 'gpu'.
    """

    results_df = pd.DataFrame(resource_log)
    sys_results = {
                   'avg_cpu_util': results_df['cpu_util'].mean(), 
                   'max_cpu_util': results_df['cpu_util'].max(),
                   'max_ram_gb': results_df['ram_gb'].max()
                  }

    if hw_type == 'gpu':
        gpu_results = {'avg_gpu_util': results_df['gpu_util'].mean(), 
                   'max_gpu_util': results_df['gpu_util'].max(),
                   'max_vram_gb': results_df['vram_gb'].max()
                  }
        sys_results = sys_results | gpu_results

    return(sys_results)

def get_telemetry(func):
    """Aquire CPU and GPU metrics during function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        resource_log = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_resources, 
                                                args=(stop_event, resource_log, hw_type))
        start_time = time.perf_counter()
        
        # Start monitoring resources
        monitor_thread.start()
    
        # Run function
        result = func(*args, **kwargs)
    
        # Signal the monitor to stop and wait for it to finish
        stop_event.set()
        elapsed_time = time.perf_counter() - start_time
        monitor_thread.join()
    
        # Process telemetry for index build stage
        telemetry = summarize_telemetry(resource_log, hw_type)
        telemetry['duration_sec'] = elapsed_time
        return (telemetry, result)
    return wrapper

@get_telemetry
def load_bin(base_path, dtype):
    """
    Load embedding vectors from a binary file written by save_bin.

    Parameters:
    - base_path: Path to file (e.g. "/output/base.fbin")
    - dtype: numpy dtype used when saving (e.g. np.float32, np.uint8, etc.)

    Returns:
    - embeddings: numpy array of shape (num_vectors, num_dimensions)
    """
    with open(base_path, 'rb') as f:
        # Read header: 2 unsigned int32, little-endian
        header = f.read(8)
        num_vectors, num_dimensions = struct.unpack('<II', header)

        # Read remaining data as flat array
        embeddings = np.fromfile(f, dtype=dtype, count=num_vectors * num_dimensions)

    return embeddings.reshape((num_vectors, num_dimensions))

@get_telemetry
def scaler_fit_transform(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return(X)

@get_telemetry
def scaler_l2_norm(X, hw_type):
    # Inplace replacement possible for numpy. Doesn't work properly for cupy.
    if hw_type == 'cpu':
        copy = False
    else:
        copy = True
        
    X = normalize(X, norm='l2', axis=1, copy=copy)
    return(X)
    
@get_telemetry
def train_kmeans(X, n_clusters, algorithm, time_delay):
    """
    Train KMeans model.
    """
    time.sleep(time_delay)
    if algorithm in ['faiss_cpu_kmeans', 'faiss_gpu_kmeans']:
        # https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
        ncentroids = n_clusters # Number of centroids
        niter = max_iter
        verbose = True
        d = X.shape[1] # Number of dimensions

        if algorithm == 'faiss_cpu_kmeans':
            use_gpu = False
        elif algorithm == 'faiss_gpu_kmeans':
            # Specify number of GPU as int, or use True to enable all GPUs.
            use_gpu = 1
            
        faiss_kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=use_gpu)
        
        # Train FAISS kmeans model:
        faiss_kmeans.train(X)
        model = faiss_kmeans
        
    elif algorithm in ['cuvs_kmeans', 'cuvs_kmeans_balanced']:
        # https://docs.rapids.ai/api/cuvs/nightly/python_api/cluster_kmeans/
        # Balanced k_means by setting hierarchical=True
        cuvs_kmeans_params = cuvs_kmeans.KMeansParams(n_clusters=n_clusters, 
                                                      max_iter=None, **cuvs_kmeans_extra_params
                                                     )
        
        # Train cuVS kmeans model:
        centroids, inertia, n_iter = cuvs_kmeans.fit(cuvs_kmeans_params, X)
        model = {'params': cuvs_kmeans_params, 'centroids': centroids, 
                 'inertia': inertia, 'n_iter': n_iter}
    else:
        raise ValueError(f"Unknown algorithm: '{algorithm}'.")
    return(model)

@get_telemetry
def predict_kmeans(model, X, algorithm, time_delay):
    """
    Apply KMeans model to assign cluster labels.
    """
    time.sleep(time_delay)
    if algorithm.startswith('faiss'):
        D, labels = model.index.search(X, 1)
        # D contains the squared L2 distances.
        
        # Flatten labels since we're only using first cluster assignments.
        labels =  labels.flatten()
        inertia = np.sum(D)
    elif algorithm.startswith('cuvs'):
        labels, inertia = cuvs_kmeans.predict(model['params'], X, model['centroids'])

        # inertia: sum of squared distances of samples to their closest cluster center

    results = {'labels': labels, 'inertia': inertia}
    return(results)

def export_results(telem_store):
    # Write results to disk
    telem_store_df = pd.DataFrame(telem_store)
    output_file = f'./results/{algorithm}_{DATASET_NAME}.csv'
    telem_store_df.to_csv(output_file, index=False)
    print()
    print(f'Output written to {output_file}')

# Start loading data
data_loading_status = 'pass'
telem_store = []
pipeline_status = {}

# Let all try - except go to "pass" to skip errors and continue grid search.

print(f'Loading dataset: {DATASET_NAME}')
if algorithm.startswith('faiss'):
    import faiss
    from sklearn.preprocessing import StandardScaler, normalize
    from sklearn.model_selection import train_test_split

    try:
        telem, X = load_bin(base_path, 'float32')
    except:
        print('** ERROR: failed to load data **')

        data_loading_status = 'fail'
        pipeline_status['ingestion'] = 'fail'
        metadata = {
            'dataset': DATASET_NAME,
            'pipeline_status': pipeline_status
        }
        
        # Store partial metadata for export
        telem_store.append(metadata)
    
        # Write results to disk
        export_results(telem_store)

        pass
        
elif algorithm.startswith('cuvs'):
    import cuml
    import cupy as cp
    from cuml.preprocessing import StandardScaler, normalize
    from cuml.model_selection import train_test_split
    import cuvs.cluster.kmeans as cuvs_kmeans
    from cuvs.cluster.kmeans import cluster_cost

    try:
        telem, X = load_bin(base_path, 'float32')
        X = cp.array(X)

        # # Check for expected normalization
        # X_row0_norm = cp.linalg.norm(X[0,:], ord=2)
        # print('Norm of first vector:', X_row0_norm)
    
        # X_col_mean = X[:,:10].mean(axis=0)
        # print('Mean of columns:', X_col_mean)
    except:
        print('** ERROR: failed to load data **')

        data_loading_status = 'fail'
        pipeline_status['ingestion'] = 'fail'
        metadata = {
            'dataset': DATASET_NAME,
            'pipeline_status': pipeline_status
        }
        
        # Store partial metadata for export
        telem_store.append(metadata)
    
        # Write results to disk
        export_results(telem_store)

        pass
        
else:
    raise ValueError(f"Unknown algorithm type: '{algorithm}'.")


# Error handling for data loading stage
pipeline_status['ingestion'] = data_loading_status

if data_loading_status == 'pass':
    telem_load_data = add_prefix(telem, 'load_data_')
    print(f'Data shape: {X.shape} \n')
    print(telem_load_data)
    print()

    # Get dataset metadata
    n_rows = len(X)
    dim = X.shape[1]
    X_dtype = str(X[0].dtype)
    
    metadata = {
        'dataset': DATASET_NAME,
        'pipeline_status': pipeline_status,
        'n_rows': n_rows,
        'dimension': dim,
        'dtype': X_dtype,
        'hw_type': hw_type,
        'algorithm': algorithm,
        'n_clusters': -1, # Temp value
        'label_counts': [], # Temp value
        'inertia_avg': -1 # Temp value
    }
    
    # Apply scaler
    if apply_scaler:
        print('Applying scaling to dataset....')
    
        try:
            # Apply standard scaler:
            # telem, X = scaler_fit_transform(X)
            # telem_scaler = add_prefix(telem, 'standard_scaler_')
        
            # Normalize L2 distance:
            telem, X = scaler_l2_norm(X, hw_type)
            telem_scaler = add_prefix(telem, 'scaler_')
            
            print(telem_scaler)
            scaler_status = 'pass'
        except:
            print('** ERROR: failed to scale data **')
            scaler_status = 'fail'
            pipeline_status['scaler'] = 'fail'
            metadata['pipeline_status'] = pipeline_status
            telem_store.append(metadata)
    
            # Write results to disk
            export_results(telem_store)
    
            pass
    else:
        telem_scaler = {}
        scaler_status = 'skip'
        
    # Update pipeline status
    pipeline_status['scaling'] = scaler_status
    metadata['pipeline_status'] = pipeline_status


    # Issue when function runs faster than polling rate (0.25s). 
    # Need to slow function execution by adding an internal delay and reversing it. 
    time_delay = 0.25
    
    print()
    print(f'Starting {algorithm} parameter sweep....')
    
    for n_clusters in k_values:
        training_status = 'pass'
        metadata['n_clusters'] = n_clusters
        
        try:
            print(f'n_clusters = {n_clusters}')
            print(f'Training {algorithm} model....')
            telem, kmeans_model = train_kmeans(X, n_clusters, algorithm, time_delay)
            telem['duration_sec'] = telem['duration_sec'] - time_delay
            telem_kmeans_train = add_prefix(telem, 'kmeans_train_')
            print(telem_kmeans_train)
            print()
    
            # Update pipeline status
            pipeline_status['training'] = training_status
            metadata['pipeline_status'] = pipeline_status
        except:
            print('** ERROR: failure encountered during model training **')
            training_status = 'fail'
    
            # Update pipeline status
            pipeline_status['training'] = training_status
            metadata['pipeline_status'] = pipeline_status
    
            # Save partial results and skip prediction step
            # telem_store.append(metadata | telem_load_data | telem_scaler)
            telem_store.append(metadata | telem_load_data)
    
            # Write results to disk
            export_results(telem_store)
    
            pass
    
        # Model prediction stage
        if training_status == 'pass':
            print(f'Predicting {algorithm} cluster labels....')
    
            try:
                telem, kmeans_result = predict_kmeans(kmeans_model, X, algorithm, time_delay)
                telem['duration_sec'] = telem['duration_sec'] - time_delay
                telem_kmeans_predict = add_prefix(telem, 'kmeans_predict_')
                print(telem_kmeans_predict)
                print()
            
                # High memory usage for computing inertia. Skip computing for now. 
                # Workaround for bug in cuvs_kmeans_balanced. Need to compute inertia externally.
                # if algorithm == 'cuvs_kmeans_balanced':
                    # kmeans_result['inertia'] = cluster_cost(X, kmeans_model['centroids'])
                    
                # Get cluster labels
                labels = kmeans_result['labels']
            
                if algorithm.startswith('cuvs'):
                    # Convert labels to numpy
                    labels = cp.asnumpy(labels)
                    
                # Count number of elements within each cluster
                label_name, label_counts = np.unique(labels, return_counts=True)
                
                # Add extra metadata from prediction stage
                metadata['label_counts'] = label_counts
                metadata['inertia_avg'] = kmeans_result['inertia'] / n_rows
    
                # Update pipeline status
                pipeline_status['prediction'] = 'pass'
                metadata['pipeline_status'] = pipeline_status
                
                # Accumulate telemetry
                telem_current = metadata | telem_load_data | telem_scaler | telem_kmeans_train | telem_kmeans_predict
                telem_store.append(telem_current)
                print(telem_current)
                print()
    
                # Write results to disk
                export_results(telem_store)
                
            except:
                print('** ERROR: failure encountered during model prediction **')
                
                # Update pipeline status
                pipeline_status['prediction'] = 'fail'
                metadata['pipeline_status'] = pipeline_status
    
                # Accumulate telemetry
                telem_current = metadata | telem_load_data | telem_scaler | telem_kmeans_train
                telem_store.append(telem_current)
    
                # Write results to disk
                export_results(telem_store)
    
                pass