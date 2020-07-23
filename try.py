import colour
from colour import read_image
import time
colour.utilities.set_ndimensional_array_backend('cupy')
RGB = read_image('testImage4K.jpg')
tim2 = time.time()
RGB2 = colour.models.RGB_to_HSL(RGB)
time2 = time.time()-tim2
print(time2)
