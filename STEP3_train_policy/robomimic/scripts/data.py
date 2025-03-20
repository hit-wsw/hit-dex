import h5py

def print_hdf5_structure(file_name):
    def print_attrs(name, obj):
        print(name)
        for key, val in obj.attrs.items():
            print(f"  Attribute: {key} => {val}")

    with h5py.File(file_name, 'r') as f:
        f.visititems(print_attrs)

# 替换成你的 HDF5 文件路径
print_hdf5_structure('/media/wsw/SSD1T/data/hand_wiping_1-14_5actiongap_10000points.hdf5')
