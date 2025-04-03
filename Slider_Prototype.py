
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Generate initial data
x = np.arange(0, 10, 0.1)
y = np.sin(x)
line, = plt.plot(x, y)

# Create the slider
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(axfreq, 'Frequency', 0.1, 10.0, valinit=1.0)

# Update the plot when the slider value changes
def update(val):
    freq = slider.val
    line.set_ydata(np.sin(freq * x))
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()