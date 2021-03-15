import numpy as np
import pandas as pd


a = "0x00"

print(int(a, 16))
print(hex(~int(a, 16)))
print(hex(~int(a, 16))[3:])

# print(a[0][0])
# print(a[0][0].dtype)
# print(a[0][0].shape)