from freeingtruth import deepl
from freeingtruth import animation
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo

d = 200              # number of features
n = 40000            # number of data samples
epochs = 5000       # number of epochs
eta = 0.01          # learning rate

w_1, w_2, w_3, w_4, loss_vals = deepl.binary_classification(d,n,epochs,eta)

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

# Create animation of 3D weights
animation.animate_weight_heatmap(
        w_1, 
        dt=0.04,
        file_name = "w1_animation",
        title_str="Evolution of $W_1$ over training epochs"
    )

animation.animate_weight_heatmap(
        w_2, 
        dt=0.04,
        file_name = "w2_animation",
        title_str="Evolution of $W_2$ over training epochs"
    )

animation.animate_weight_heatmap(
        w_3, 
        dt=0.04,
        file_name = "w3_animation",
        title_str="Evolution of $W_3$ over training epochs"
    )

animation.animate_weight_heatmap(
        w_4, 
        dt=0.04,
        file_name = "w4_animation",
        title_str="Evolution of $W_4$ over training epochs"
    )