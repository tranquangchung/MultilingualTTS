import argparse

import yaml

# from preprocessor.preprocessor import Preprocessor
from preprocessor.preprocessor_multispeaker import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
