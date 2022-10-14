import configparser


def get_configs(configs_path="configs.cfg"):
    """"""
    config = configparser.ConfigParser()
    config.read(configs_path)

    return config
