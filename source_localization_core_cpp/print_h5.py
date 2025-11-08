import h5py
import sys

def list_hdf5_items(filename):
    def print_name(name):
        print(name)

    try:
        with h5py.File(filename, 'r') as f:
            print(f"\nContents of '{filename}':")
            f.visit(print_name)
    except OSError as e:
        print(f"Error opening file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_hdf5_items.py <filename.h5>")
    else:
        list_hdf5_items(sys.argv[1])