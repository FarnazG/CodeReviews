import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sampler as sm


def target(sample):
    # target_dis = stats.beta(a=37.0, b=25)
    target_dis = stats.gamma(a=1.99)
    return target_dis.pdf(sample)


def target2d(sample):
    target_dis = stats.gamma(a=3.99)
    return target_dis.pdf(sample[0,]) * target_dis.pdf(sample[1,])


