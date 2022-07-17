import pathlib
import logging

import torch


LOGGER = logging.getLogger(__name__)


def get_profiler(device: torch.device, log_dir: pathlib.Path, run_id: str) -> torch.profiler.profile:
    profile_activity = torch.profiler.ProfilerActivity.CPU
    log_into = log_dir / f"profile-{run_id}"
    if device == torch.device("cuda"):
        profile_activity = torch.profiler.ProfilerActivity.CUDA

    LOGGER.info("Profile %s activity into %s", device, log_into)

    return torch.profiler.profile(
        activities=[profile_activity],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_into)),
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
        with_stack=True
    )
