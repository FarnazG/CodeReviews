import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sampler as sm


def target(sample):
    # target_dis = stats.beta(a=37.0, b=25)
    target_dis = stats.gamma(a=1.99)
    return target_dis.pdf(sample)


def target2d(sample):
    target_dis = stats.gamma(a=3.99).pdf(sample[1,])
    normal_dis = stats.multivariate_normal(0, 1).pdf(sample[0,])
    return target_dis * normal_dis
