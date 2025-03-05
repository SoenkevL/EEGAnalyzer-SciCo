# %% imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal

# %% create simple mock signals
fs = 100
dt = 1 / fs
T = 20
t = np.arange(0, T, dt)

# some noisy signals
noise1 = np.random.randn(len(t))
noise2 = np.random.randn(len(t)) * 5
normal1 = np.random.normal(0, 1, len(t))
normal2 = np.random.normal(1, 1, len(t))
normal3 = np.random.normal(1, 2, len(t))
exp_dist1 = np.random.exponential(1, len(t))

# constants
zeros1 = np.zeros(len(t))
ones1 = np.ones(len(t))

# some simple sines
sin1 = np.sin(2 * np.pi * t)
sin2 = np.sin(4 * np.pi * t) + sin1
sin3 = np.sin(8 * np.pi * t) + np.cos(12 * np.pi * t) + sin2


# linear funcs
def linear_func(t, m, b):
    return m * t + b


linear1 = linear_func(t, 0.1, 0)
linear2 = linear_func(t, 0.5, 0)
linear3 = linear_func(t, 2, 0)

# other
sawtooth1 = signal.sawtooth(2 * np.pi * 2 * t)
sawtooth2 = signal.sawtooth(2 * np.pi * 6 * t, width=0.5)
sawtooth3 = signal.sawtooth(2 * np.pi * 4 * t, width=0)
square1 = signal.square(2 * np.pi * 2 * t)
square2 = signal.square(2 * np.pi * 6 * t, duty=0.25)
square3 = signal.square(2 * np.pi * 4 * t, duty=0.65)

# sines + linear
linear1_sin1 = linear1 + sin1
linear2_sin2 = linear2 + sin2
linear3_sin3 = linear3 + sin3

# other + linear
sawtooth1_linear1 = sawtooth1 + linear1
sawtooth2_linear3 = sawtooth2 + linear3
square2_linear2 = square2 + linear2
square3_linear3 = square3 + linear3

# added_noise
sin3_noise1 = sin3 + noise1
sin3_normal3 = sin3 + normal3
sin3_noise2 = sin3 + noise2
sin3_exp_dist1 = sin3 + exp_dist1

linear1_sin1_noise1 = linear1_sin1 + noise1
linear2_sin2_normal2 = linear2_sin2 + normal2
linear3_sin3_noise2 = linear3_sin3 + noise2

# %% plotting of the functions
first_5_t = t[:5 * fs]
first_5_t_idx = np.arange(len(first_5_t))

fig, ax = plt.subplots(1, 1)
ax.plot(first_5_t, np.transpose([sin1[first_5_t_idx], sin2[first_5_t_idx], sin3[first_5_t_idx]]),
        label=['sin1', 'sin2', 'sin3'])
fig.suptitle('sin functions')
fig.legend()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(first_5_t, np.transpose([linear1[first_5_t_idx], linear2[first_5_t_idx], linear3[first_5_t_idx]]),
        label=['linear1', 'linear2', 'linear3'])
fig.suptitle('linear functions')
fig.legend()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(first_5_t, np.transpose(
    [noise1[first_5_t_idx], noise2[first_5_t_idx], normal1[first_5_t_idx], normal2[first_5_t_idx],
     normal3[first_5_t_idx], exp_dist1[first_5_t_idx]]),
        label=['noise1', 'noise2', 'normal1', 'normal2', 'normal3', 'exp_dist1'])
fig.suptitle('noise functions')
fig.legend()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(first_5_t, np.transpose(
    [sawtooth1[first_5_t_idx], sawtooth2[first_5_t_idx], sawtooth3[first_5_t_idx], square1[first_5_t_idx],
     square2[first_5_t_idx], square3[first_5_t_idx]]),
        label=['sawtooth1', 'sawtooth2', 'sawtooth3', 'square1', 'square2', 'square3'])
fig.suptitle('other periodic waves')
fig.legend()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(first_5_t, np.transpose(
    [sawtooth1_linear1[first_5_t_idx], sawtooth2_linear3[first_5_t_idx], square2_linear2[first_5_t_idx],
     square3_linear3[first_5_t_idx]]),
        label=['sawtooth1_linear1', 'sawtooth2_linear3', 'square2_linear2', 'square3_linear3'])
fig.suptitle('linear combinations')
fig.legend()
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(first_5_t, np.transpose([sin3_noise1[first_5_t_idx], sin3_normal3[first_5_t_idx], sin3_noise2[first_5_t_idx],
                                 sin3_exp_dist1[first_5_t_idx]]),
        label=['sin3_noise1', 'sin3_normal3', 'sin3_noise2', 'sin3_exp_dist1'])
fig.legend()
fig.suptitle('sin3 with different noise functions')
plt.show()

fig, ax = plt.subplots(1, 1)
ax.plot(first_5_t, np.transpose(
    [linear1_sin1_noise1[first_5_t_idx], linear2_sin2_normal2[first_5_t_idx], linear3_sin3_noise2[first_5_t_idx]]),
        label=['linear1_sin1_noise1', 'linear2_sin2_normal2', 'linear3_sin3_noise2'])
fig.suptitle('linear_sin with different noise functions')
fig.legend()
plt.show()

# %% create a dataframe
headers = ['constant_0', 'constant_1', 'sin1', 'sin2', 'sin3', 'linear1', 'linear2', 'linear3', 'noise1', 'noise2', 'normal1', 'normal2', 'normal3',
           'exp_dist1', 'sawtooth1', 'sawtooth2', 'sawtooth3', 'square1', 'square2', 'square3', 'sawtooth1_linear1',
           'sawtooth2_linear3', 'square2_linear2', 'square3_linear3', 'sin3_noise1', 'sin3_normal3', 'sin3_noise2',
           'sin3_exp_dist1', 'linear1_sin1_noise1', 'linear2_sin2_normal2', 'linear3_sin3_noise2']
data = np.transpose(
    [zeros1, ones1, sin1, sin2, sin3, linear1, linear2, linear3, noise1, noise2, normal1, normal2, normal3, exp_dist1, sawtooth1,
     sawtooth2, sawtooth3, square1, square2, square3, sawtooth1_linear1, sawtooth2_linear3, square2_linear2,
     square3_linear3, sin3_noise1, sin3_normal3, sin3_noise2, sin3_exp_dist1, linear1_sin1_noise1, linear2_sin2_normal2,
     linear3_sin3_noise2])
export_frame = pd.DataFrame(data, columns=headers)
export_frame.to_csv(
    '/home/soenkevanloh/Documents/EEGAnalyzer/Artificial_signal_tests/csv/simple_mock_signal_frame.csv',
    index=True, header=True)
