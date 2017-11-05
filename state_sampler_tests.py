# Not formal tests
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from lib.state_sampler import StateSampler

def test2Dsampler():
    s = StateSampler()
    for i in range(10):
        print(s.getRandom2DState())


def test3Dsampler():
    s = StateSampler()
    for i in range(10):
        print(s.getRandom3DState())


if __name__ == "__main__":
    test2Dsampler()
    test3Dsampler()