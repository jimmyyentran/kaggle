from PIL import Image
import os.path

filename = os.path.join('images', 'processed_manual')
mini = 1000
maxi = 0
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

print mini
print maxi
