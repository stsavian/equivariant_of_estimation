import argparse

def main(args):


    return
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--chairs_pth', type=str, nargs='+')
    parser.add_argument('--chairs2_pth', type=str, nargs='+')
    parser.add_argument('--chairsOcc_pth', type=str, nargs='+')
    parser.add_argument('--things_pth', type=str, nargs='+')
    parser.add_argument('--monkaa_pth', type=str, nargs='+')
    parser.add_argument('--hd1k_pth', type=str, nargs='+')
    parser.add_argument('--kitti_pth', type=str, nargs='+')
    parser.add_argument('--matlab_pth', type=str, nargs='+')
    parser.add_argument('--matlab_equiv_pth', type=str, nargs='+')
    parser.add_argument('--sintel_pth', type=str, nargs='+')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[0,5,20, 10000])
    parser.add_argument('--results_pth', type=str, nargs='+')

    args = parser.parse_args()
    thresholds = args.thresholds
