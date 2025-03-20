import json


REGISTERED_CONFIGS = {}
def config_factory(algo_name, dic=None):
    """
    Creates an instance of a config from the algo name. Optionally pass
    a dictionary to instantiate the config from the dictionary.
    """
    if algo_name not in REGISTERED_CONFIGS:
        raise Exception("Config for algo name {} not found. Make sure it is a registered config among: {}".format(
            algo_name, ', '.join(REGISTERED_CONFIGS)))
    return REGISTERED_CONFIGS[algo_name](dict_to_load=dic)

ext_cfg = json.load(open('E:\wsw\python\DexCap-main\STEP3_train_policy\\robomimic\\training_config\diffusion_policy_pcd_wiping_1-14.json', 'r'))
config = config_factory(ext_cfg["algo_name"])
print(config)
