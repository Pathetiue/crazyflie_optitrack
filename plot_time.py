import numpy as np
import matplotlib.pyplot as plt

data = np.load("./data/training_data/control5000.npy")
print(data.shape)
timestamp = data[:]
period = timestamp[1:] - timestamp[:(-1)]

plt.plot(period)
plt.show()