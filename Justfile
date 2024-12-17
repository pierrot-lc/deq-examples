platform:
    TF_CPP_MIN_LOG_LEVEL=0 python -c "from jax.extend.backend import get_backend; print(get_backend().platform)"

download-mnist:
    kaggle datasets download -d ben519/mnist-as-png
    unzip ./mnist-as-png.zip
    mkdir -p data
    rm -rf data/mnist
    mkdir data/mnist
    mv ./mnist-png/* ./data/mnist
    rm -rf ./mnist-as-png.zip ./mnist-png

tests:
    JAX_PLATFORM_NAME=cpu python3 -m pytest -v tests/

# Pip-related.
compile-requirements:
    uv pip freeze | uv pip compile - -o requirements.txt
