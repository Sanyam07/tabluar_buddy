# @Author: yican.kz
# @Date: 2019-08-18 23:45:03
# @Last Modified by:   yican.kz
# @Last Modified time: 2019-08-18 23:45:03

# Standard libraries
import os

# Third party libraries
import glob


def mkdir(path: str):
    """Create directory.

     Create directory if it is not exist, else do nothing.

     Parameters
     ----------
     path: str
        Path of your directory.

     Examples
     --------
     mkdir("data/raw/train/")
     """
    try:
        os.stat(path)
    except Exception:
        os.makedirs(path)


def remove_temporary_files(folder_path: str):
    """Remove files begin with ".~".

     Parameters
     ----------
     folder_path: str
        Folder path which you want to clean.

     Examples
     --------
     remove_temporary_files("data/raw/")

    """
    num_of_removed_file = 0
    for fname in os.listdir(folder_path):
        if fname.startswith("~") or fname.startswith("."):
            num_of_removed_file += 1
            os.remove(folder_path + "/" + fname)
    print("{0} file have been removed".format(num_of_removed_file))


def remove_all_files(folder_path: str):
    """Remove all files under folder_path.

     Parameters
     ----------
     folder_path: str
        Folder path which you want to clean.

     Examples
     --------
     remove_all_files("data/raw/")

    """
    folder = folder_path + "*"
    files = glob.glob(folder)
    for file in files:
        os.remove(file)


def save_last_n_files(directory, max_to_keep=10, suffix="*.p"):
    """Save max_to_keep files with suffix specified in directory

     Parameters
     ----------
     directory: str
        Folder path which you save files.
     max_to_keep: int
        Maximum number of files to keep.
     suffix: str
        File suffix.

     Examples
     --------
     save_last_n_files("data/raw/")
    """
    saved_model_files = glob.glob(directory + suffix)
    saved_model_files_lasted_n = sorted(saved_model_files, key=os.path.getctime)[-max_to_keep:]
    files_tobe_deleted = set(saved_model_files).difference(saved_model_files_lasted_n)

    for file in files_tobe_deleted:
        os.remove(file)
