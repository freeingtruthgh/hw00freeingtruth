from freeingtruth.deep1.two_layer_binary_classification import binary_classification
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo

d = 64              # number of features
n = 1000            # number of data samples
epochs = 10000      # default value of epochs

w_1, w_2, w_3, w_4, loss_vals = binary_classification(d,n)

# Create a time stamp to name the file
filename = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y%m%d%H%M%S.pdf")
print(filename)

# plot loss vs epochs
plt.plot(range(epochs),loss_vals,linewidth=2)
# create title, and x, y labels
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(filename)    # save PDF of plot
plt.show()

