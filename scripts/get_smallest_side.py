from PIL import Image
import os.path
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

filename = os.path.join('images', 'processed_manual')
#  filename = os.path.join('images', 'processed_2')
mini = 1000
maxi = 0
sides= []
areas = []
for i in os.listdir(filename):
    img = Image.open(os.path.join(filename, i))
    width, height = img.size
    if width < mini:
        mini = width
    if width > maxi:
        maxi = width
    if height < mini:
        mini = height
    if height > maxi:
        mini = height

    sides.append(width)
    sides.append(height)

    areas.append(width * height)

print mini
print maxi

N = len(os.listdir(filename))
n = N/10

#  smooth1 = 6000
#  p, x = np.histogram(sides, bins=n)
#  x = x[:-1] + (x[1] - x[0])/2
#  f = UnivariateSpline(x, p, s=smooth1)
#  plt.plot(x, f(x))

# 350 seems to be a good number
plt.subplot(121)
plt.hist(sides, 40, normed=1, facecolor='green')
plt.grid(True)

plt.subplot(122)
plt.hist(areas, 40, normed=1, facecolor='green')
plt.grid(True)
plt.show()
