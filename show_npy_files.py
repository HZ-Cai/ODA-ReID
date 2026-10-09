import numpy as np

# npy file path
npy_path = ""

# load file
data = np.load(npy_path)

print("=" * 60)
print("NPY FILE INFO")
print("=" * 60)

# basic information
print(f"File Path        : {npy_path}")
print(f"Type             : {type(data)}")
print(f"Shape            : {data.shape}")
print(f"Dtype            : {data.dtype}")
print(f"Dimension        : {data.ndim}")

print()

# value range
print("=" * 60)
print("VALUE RANGE")
print("=" * 60)

print(f"Min Value        : {data.min()}")
print(f"Max Value        : {data.max()}")
print(f"Mean Value       : {data.mean()}")
print(f"Std Value        : {data.std()}")

print()

# check for NaN / Inf
print("=" * 60)
print("SPECIAL VALUES")
print("=" * 60)

print(f"Contains NaN     : {np.isnan(data).any()}")
print(f"Contains Inf     : {np.isinf(data).any()}")

print()

print(data.shape)

# if data.ndim == 3:
#     print("=" * 60)
#     print("CHANNEL SHAPES")
#     print("=" * 60)

#     for i in range(data.shape[0]):
#         print(f"Channel {i}: {data[i].shape}")

# print()

if data.ndim == 3:
    print("=" * 60)
    print("FORMAT GUESS")
    print("=" * 60)

    if data.shape[0] <= 30:
        print(f"Possibly CHW format : {data.shape}")

    if data.shape[-1] <= 30:
        print(f"Possibly HWC format : {data.shape}")

print()

print("=" * 60)
print("DONE")
print("=" * 60)