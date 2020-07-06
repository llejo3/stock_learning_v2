import logging.config
import yaml
import os


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dir_path, "logging.yaml"), 'r') as f:
        config = yaml.safe_load(f.read())
        for handler_name in config["handlers"]:
            handler = config["handlers"][handler_name]
            if "filename" in handler:
                handler["filename"] = handler["filename"].format(path=dir_path)
        logging.config.dictConfig(config)


def get_logger(name: str = None):
    return logging.getLogger(name)


main()
