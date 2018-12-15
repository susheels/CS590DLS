import split_folders

# split with a ratio. To only split into training and validation set, set a tuple, e.g. (.8, .2)

file_dir_path = "llvm_ir/"
split_folders.ratio(file_dir_path, output="", seed=1337, ratio=(.6, .2, .2)) # default values
