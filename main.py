import warnings
warnings.filterwarnings("ignore")

import scenario
import util
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    args = util.get_args()
    method = getattr(scenario, args.scenario)
    method(args)

if __name__ == '__main__':
    main()

# 00%|█| 500/500 [08:00<00:00,  1.04it/s, estoi:enhanced=0.653, pesq:enhanced=1.85, si_snr:enhanced=11.6, si_snr:noisy=5.94, stoi:enhanc