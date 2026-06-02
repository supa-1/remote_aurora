#!/usr/bin/env bash

setup_hpc_env() {
    local env_name="${1:-reconvla}"

    if [[ -f /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
    fi

    if [[ -f /home/HPCBase/tools/module-5.2.0/init/profile.sh ]]; then
        # shellcheck disable=SC1091
        source /home/HPCBase/tools/module-5.2.0/init/profile.sh
        module use /home/HPCBase/modulefiles/ || true
        module load compilers/nvhpc_sdk/23.5_cuda_11.8_12.1 || true
        module load libs/cudnn/8.9.5_cuda12 || true
    fi

    if command -v conda >/dev/null 2>&1; then
        if [[ "${CONDA_DEFAULT_ENV:-}" != "$env_name" ]]; then
            conda activate "$env_name"
        fi
    fi
}
