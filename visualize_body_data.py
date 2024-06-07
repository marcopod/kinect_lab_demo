import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load your JSON data
with open("./skeleton_data_2024-06-06T18-45-48.json", "r") as file:
    data = json.load(file)


# Assuming each item in `data` corresponds to a frame of tracking data
def update_graph(num):
    graph._offsets3d = (
        [
            data[num]["joints"][j]["position"]["x"]
            for j in range(len(data[num]["joints"]))
        ],
        [
            data[num]["joints"][j]["position"]["y"]
            for j in range(len(data[num]["joints"]))
        ],
        [
            data[num]["joints"][j]["position"]["z"]
            for j in range(len(data[num]["joints"]))
        ],
    )
    title.set_text("3D Test, time={}".format(data[num]["timestamp"]))


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
title = ax.set_title("3D Body Tracking")

# Initial positions of joints
x = [data[0]["joints"][j]["position"]["x"] for j in range(len(data[0]["joints"]))]
y = [data[0]["joints"][j]["position"]["y"] for j in range(len(data[0]["joints"]))]
z = [data[0]["joints"][j]["position"]["z"] for j in range(len(data[0]["joints"]))]

graph = ax.scatter(x, y, z)

# Creating the animation
ani = FuncAnimation(fig, update_graph, frames=len(data), interval=100, repeat=True)

plt.show()
