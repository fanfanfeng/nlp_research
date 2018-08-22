# create by fanfan on 2018/7/17 0017
import yaml
from seq2seq import setting

def load_parameter(path):
    with open(path,encoding='utf-8') as fread:
        params = yaml.load(fread)

    return params

if __name__ == '__main__':
    params = load_parameter(setting.yaml_file)
    print(params['optimizer']['learning_rate'])