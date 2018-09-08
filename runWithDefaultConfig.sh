pushd `dirname "$0"` > /dev/null
python3 src/interface.py --config example_config.yaml
popd > /dev/null
