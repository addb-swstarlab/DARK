import argparse
import re
import os

parser = argparse.ArgumentParser()
parser.add_argument('--log', type= str, required=True)

args = parser.parse_args()

if __name__ == '__main__':
    f = open(os.path.join('./logs',f'log-{args.log}.log'))
    for data in f.readlines():
        if 'RDB' or 'AOF' in data:
            if 'RDB' in data:
                persistence = 'RDB'
            elif 'AOF' in data:
                persistence = 'AOF'
        if 'Target' in data:
            target = data.strip().split()[-1]
        if 'Knob save path' in data:
            save_path = data.strip().split()[-1].split('/')[-1]
        if 'Choose Top' in data:
            topk = re.findall("\d+",data.split("[")[1])[-1]
        if 'Model save path' in data:
            model_save_path = data.strip().split()[-1].split('/')[-1]
        if 'Best Epoch' in data:
            num = re.findall("\d+",data)[0]
        if 'Pre-training Stage' in data:
            if 'single' in data:
                model_mode = 'single'
            elif 'twice' in data:
                model_mode = 'twice'

    print(f'python GA.py --target {target} --persistence {persistence} --topk {topk} --sk {save_path} --path {model_save_path} --num {num} --model_mode {model_mode}')