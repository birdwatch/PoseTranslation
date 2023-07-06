import subprocess
from logging import getLogger

import torch

logger = getLogger(__name__)

DEFAULT_ATTRIBUTES = (
    "index",
    "uuid",
    "name",
    "timestamp",
    "memory.total",
    "memory.free",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
)


def get_gpu_utilization(nvidia_smi_path: str = "nvidia-smi", keys=DEFAULT_ATTRIBUTES, no_units: bool = True):
    nu_opt = "" if not no_units else ",nounits"
    cmd = "%s --query-gpu=%s --format=csv,noheader%s" % (
        nvidia_smi_path,
        ",".join(keys),
        nu_opt,
    )
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]

    gpu_info = [{k: v for k, v in zip(keys, line.split(", "))} for line in lines]
    result = []
    for info in gpu_info:
        result.append(
            [
                int(info["index"]),
                float(info["memory.used"]) / float(info["memory.total"]),
            ]
        )
    return result


def get_device(allow_only_gpu: bool = True) -> str:
    if torch.cuda.is_available():
        gpu_utilization = get_gpu_utilization()
        gpu_utilization = sorted(gpu_utilization, key=lambda x: x[1])

        free_gpu_id = gpu_utilization[0][0]
        device = f"cuda:{free_gpu_id}"
        torch.backends.cudnn.benchmark = True
    else:
        if allow_only_gpu:
            message = "You can use only cpu while you don't" "allow the use of cpu alone during training."
            logger.error(message)
            raise ValueError(message)

        device = "cpu"
        logger.warning(
            "CPU will be used for training. It is better to use GPUs instead"
            "because training CNN is computationally expensive."
        )

    logger.info(f"device: {device}")

    return device
