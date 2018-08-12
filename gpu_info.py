
import subprocess

def get_gpus():
    """Return list of GPUs on the machine.

    Returns:
        gpu_list: List of (GPU_UUID, GPU_INDEX)
    """    

    # get list of gpus (index, uuid)
    list_gpus = subprocess.check_output(["nvidia-smi", "--list-gpus"]).decode()
    # parse index and guid
    gpus = [x for x in list_gpus.split('\n') if len(x) > 0]

    def parse_gpu_str(gpu_str):
        cols = gpu_str.split(' ')
        return cols[5].split(')')[0], cols[1].split(':')[0]

    gpu_list = [parse_gpu_str(gpu) for gpu in gpus]
    return gpu_list
