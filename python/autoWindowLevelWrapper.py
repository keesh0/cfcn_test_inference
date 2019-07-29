from ctypes import cdll
lib = cdll.LoadLibrary('./libautowindowlevel.so')

Window = ctypes.c_double()
Level = ctypes.c_double()

width = ctypes.c_int(width_from_image)
height = ctypes.c_int(height_from_image)

HasPadding = ctypes.c_bool(False)
PaddingValue = ctypes.c_int(0)

Slope = ctypes.c_double(slope_from_image)
Intercept = ctypes.c_double(intercept_from_image)

c_int_p = ctypes.POINTER(ctypes.c_int)
image_data = image_data.astype(numpy.int16)
data = image_data.ctypes.data_as(c_int_p)

# 888 LEFT OFF HERE need to convert from numpy to ctypes ptr
lib.AutoWindowLevel(data, width, height, Intercept, Slope, HasPadding, PaddingValue, ctypes.byref(Window), ctypes.byref(Level))
win = Window.value
lev = Level.value
print("Auto W/L window = " + str(win) + ", level = " + str(lev))