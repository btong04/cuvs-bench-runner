import os
import glob
import numpy as np
import struct
import pandas as pd
import threading
import subprocess
import time
from tqdm import tqdm

def monitor_resources(stop_event, log, hw_type='cpu'):
    """
    Monitor CPU and RAM usage every 1 second until stop_event is set.
    """
    import psutil
    
    if hw_type == 'gpu':
        import pynvml
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Get device handle for GPU_0
        
    while not stop_event.is_set():
        cpu_util = psutil.cpu_percent(interval=1)  # Measures over 1 second
        ram_util = psutil.virtual_memory().used/(1024**3)

        sys_util = {'timestamp': time.perf_counter(), 'cpu_util': cpu_util, 'ram_gb': ram_util}

        if hw_type == 'gpu':
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_mem_used_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).used/(1024**3)
            gpu_util = {'gpu_util': gpu_utilization, 'vram_gb': gpu_mem_used_gb}

            # Append GPU telemetry to system telemetry
            sys_util = sys_util | gpu_util

        # Store system logs
        log.append(sys_util)

    if hw_type == 'gpu':
        pynvml.nvmlShutdown()

def summarize_telemetry(resource_log, stage_prefix, hw_type='cpu'):
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
    sys_results = {stage_prefix + '_duration_sec': results_df['timestamp'].max() - results_df['timestamp'].min(),
                   stage_prefix + '_avg_cpu_util': results_df['cpu_util'].mean(), 
                   stage_prefix + '_max_cpu_util': results_df['cpu_util'].max(),
                   stage_prefix + '_max_ram_gb': results_df['ram_gb'].max()
                  }

    # Note: duration include overhead of starting docker. Look inside ./result/build dir
    # for more accurate timing.

    if hw_type == 'gpu':
        gpu_results = {stage_prefix + '_avg_gpu_util': results_df['gpu_util'].mean(), 
                   stage_prefix + '_max_gpu_util': results_df['gpu_util'].max(),
                   stage_prefix + '_max_vram_gb': results_df['vram_gb'].max()
                  }
        sys_results = sys_results | gpu_results

    return(sys_results)

def run_command_with_telemetry(bash_command, stage_prefix, hw_type='cpu'):
    """
    Wrapper function to run commands in bash with python monitoring system resources.
    """
    resource_log = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, 
                                            args=(stop_event, resource_log, hw_type))

    # Start monitoring resources
    monitor_thread.start()

    # Run function
    subprocess.run(bash_command, shell=True)

    # Signal the monitor to stop and wait for it to finish
    stop_event.set()
    monitor_thread.join()

    # Process telemetry for index build stage
    telemetry = summarize_telemetry(resource_log, stage_prefix, hw_type)
    return(telemetry)

def generate_cuvs_bench_run_cmd(DATASET_NAME, ALGORITHM, K, BATCH_SIZE, GROUP, SEARCH_MODE, SEARCH_ONLY=False):
    run_cmd = f"""
    conda run -n base python -m cuvs_bench.run \
    -k {K} \
    -bs {BATCH_SIZE} \
    --algorithms {ALGORITHM} \
    --groups {GROUP} -f \
    -m {SEARCH_MODE} \
    --dataset-configuration /data/benchmarks/datasets/{DATASET_NAME}/config.yaml \
    --dataset {DATASET_NAME} \
    --dataset-path /data/benchmarks/datasets \
    --configuration /data/benchmarks/configs/{ALGORITHM}.yaml """

    if SEARCH_ONLY:
        run_cmd = run_cmd + '--search'
        
    return(run_cmd)