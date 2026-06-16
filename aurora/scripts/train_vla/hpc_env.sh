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

    # Make conda-provided shared libraries visible to Python extension modules
    # launched from DSUB's non-interactive shell. This covers runtimes such as
    # libopenblas.so.0 that torch/transformers may need before Python code runs.
    if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$CONDA_PREFIX/lib:"*) ;;
            *) export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi

    # On the ARM A100 server, sklearn may load its bundled libgomp too late and
    # fail with "cannot allocate memory in static TLS block". Preload libgomp
    # before Python starts so transformers.trainer can import sklearn safely.
    local libgomp_path=""
    if [[ -n "${CONDA_PREFIX:-}" && -f "$CONDA_PREFIX/lib/libgomp.so.1" ]]; then
        libgomp_path="$CONDA_PREFIX/lib/libgomp.so.1"
    elif [[ -n "${CONDA_PREFIX:-}" ]]; then
        libgomp_path="$(find "$CONDA_PREFIX/lib/python"*"/site-packages/scikit_learn.libs" -name 'libgomp*.so*' 2>/dev/null | head -n 1 || true)"
    fi

    if [[ -n "$libgomp_path" && -f "$libgomp_path" ]]; then
        case ":${LD_PRELOAD:-}:" in
            *":$libgomp_path:"*) ;;
            *) export LD_PRELOAD="$libgomp_path${LD_PRELOAD:+:$LD_PRELOAD}" ;;
        esac
    fi
}
