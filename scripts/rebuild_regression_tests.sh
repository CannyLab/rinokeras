
# For each version in the code
declare -a TF_VERSIONS=("1.12" "1.13" "2.0.0a0")
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

for ver in "${TF_VERSIONS[@]}"; do
    # Create and activate the virtual environment
    virtualenv .tmp_env --python=python3
    source .tmp_env/bin/activate

    # Install the requirements for testing
    pip install "tensorflow==$ver"
    pip install -r $DIR/../requirements.txt
    
    # Extra requirements
    pip install nltk
    pip install tensorflow_datasets
    
    # Install the library
    pip install -e $DIR/../

    # Run/rebuild the tests
    export RK_REBUILD_REGRESSION_TESTS=True
    pytest $DIR/../

    # Remove the virutal environment
    deactivate
    rm -r .tmp_env
done

