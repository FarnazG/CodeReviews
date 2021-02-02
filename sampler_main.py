import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sampler as sm


def target(sample):
    # target_dis = stats.beta(a=37.0, b=25)
    target_dis = stats.gamma(a=1.99)
    return target_dis.pdf(sample)


#def target2d(sample):
#    target_dis = stats.gamma(a=3.99)
#    return target_dis.pdf(sample[0,]) * target_dis.pdf(sample[1,])

def target2d(sample):
    target_dis = stats.gamma(a=3.99).pdf(sample[1,])
    normal_dis = stats.multivariate_normal(0, 1).pdf(sample[0,])
    return target_dis * normal_dis


dim = 1
if dim == 1:
    sams = np.linspace(-5, 5, 500)
    plt.plot(sams, target(sams), c='orange')
    samples = sm.Sampler(dim=dim).metropolis_hastings(target=target, niter=2500, nburn=100)
    plt.hist(samples[0,], bins=100, density=True, linewidth=1)
    plt.show()
elif dim == 2:
    X, Y = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    Z = np.array([X, Y])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].contourf(X, Y, target2d(Z))
    samples = sm.Sampler(dim=dim).metropolis_hastings(target=target2d, start=-0.5, niter=6000, nburn=100)
    ax[1].hist2d(samples[0,], samples[1,], bins=100, density=True, linewidth=1, range=[[-5, 5], [-5, 5]])
    plt.show()

dim = 0
if dim == 1:
    sams = np.linspace(-5, 5, 500)
    plt.plot(sams, target(sams), c='orange')
elif dim == 2:
    X, Y = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    Z = np.array([X, Y])
    samples, weights = sm.Sampler(dim=dim, coef=20.0).importance(target=target2d, niter=6000, nburn=100)
    ax[1].hist2d(samples[0,], samples[1,], bins=100, weights=weights, linewidth=1, range=[[-5, 5], [-5, 5]])
    plt.show()
