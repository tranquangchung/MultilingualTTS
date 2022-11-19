import argparse
import os
import yaml

from preprocessor.preprocessor_multi import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    ### check metadata.csv #########
    out_dir = config["path"]["preprocessed_path"]
    metadata_path = os.path.join(out_dir, "metadata.csv")
    if not os.path.isfile(metadata_path):
        preprocessor.write_metadata()
    #################################
    preprocessor.build_from_path_paralel()
