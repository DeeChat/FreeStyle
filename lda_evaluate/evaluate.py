import math
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='lda_train.dat.theta')
args = parser.parse_args()

def main():
    ppls = []
    with open(args.input_file) as f:
        for line in f.readlines():
            probs = list(map(float, line.split()))
            ppl = sum(map(lambda x: - math.log(abs(x)), probs)) / len(probs)
            ppl = 2 ** ppl
            # print('ppl: ', ppl)
            ppls.append(ppl)

    with open(args.input_file + '.eval', 'w') as f:
        for ppl in ppls:
            f.write('{:0.5f}\n'.format(ppl))
        f.write('Mean: {:0.5f}\n'.format(np.mean(ppls)))

    print('Mean: ', np.mean(ppls))

if __name__ == '__main__':
    '''
    usage:
    python evaluate.py --input_file lda_netease_models/gan_answer.dat.theta
    '''
    main()
