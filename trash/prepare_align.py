import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, vntts, vntts_lien, libritts_for_VN, vntts_vananh, vntts_doanduylinh, vntts_nga, vntts_hoangnhan


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    # if "VNTTS" in config["dataset"]:
    #     vntts.prepare_align(config)
    if "VNTTS_LIEN" in config["dataset"]:
        vntts_lien.prepare_align(config)
    if "VNTTS_MultiSpeaker" in config["dataset"]:
        libritts_for_VN.prepare_align(config)
    if "VNTTS_VANANH" in config["dataset"]:
        vntts_vananh.prepare_align(config)
    if "VNTTS_doanduylinh" in config["dataset"]:
        vntts_doanduylinh.prepare_align(config)
    if "VNTTS_NGA" in config["dataset"]:
        vntts_nga.prepare_align(config)
    if "VNTTS_Hoangnhan" in config["dataset"]:
        vntts_hoangnhan.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
