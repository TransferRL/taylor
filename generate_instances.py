# Not formal tests
import sys
import os
import pickle as cPickcle

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from lib.instance_sampler import InstanceSampler

def generate2Dsampler(num):
    s = InstanceSampler()
    result = [];
    for i in range(num):
        result.append(s.getRandom2DInstance())
    return result


def generate3Dsampler(num):
    s = InstanceSampler()
    result = [];
    for i in range(num):
        result.append(s.getRandom3DInstance())
    return result


if __name__ == "__main__":
    # with open('./data/2d_instances.pkl', "wb+") as f:
    #     cPickcle.dump(generate2Dsampler(5000), f)
    # with open('./data/3d_instances.pkl', "wb+") as f:
    #     cPickcle.dump(generate3Dsampler(5000), f)

    # Read example
    # Data format:
    # Arrays of instances. Each instance:
    # [state, action, next_state, reward, done]
    with open('./data/2d_instances.pkl', "rb") as f:
        instances = cPickcle.load(f)
        print(instances[:5])
    with open('./data/3d_instances.pkl', "rb") as f:
        instances = cPickcle.load(f)
        print(instances[:5])