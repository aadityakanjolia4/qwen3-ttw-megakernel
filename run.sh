#!/usr/bin/env bash
set -e

CUBLAS_LIB=$(python -c "import nvidia.cublas, os; print(os.path.dirname(nvidia.cublas.__file__))")/lib
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))")/lib

export LD_LIBRARY_PATH="$CUBLAS_LIB:$CUDNN_LIB:$LD_LIBRARY_PATH"

exec "$@"
