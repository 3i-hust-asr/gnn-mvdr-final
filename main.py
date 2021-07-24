import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import scenario
import util

def main():
    args = util.get_args()
    method = getattr(scenario, args.scenario)
    method(args)

if __name__ == '__main__':
    main()