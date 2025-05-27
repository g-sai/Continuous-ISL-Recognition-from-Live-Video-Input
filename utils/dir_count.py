import os

def count_files_in_directory(directory):
    
    """
    Counts and prints the number of files in each subdirectory within the given directory.

    Args:
        directory (str): The path to the directory to scan.

    """

    for root, dirs, files in os.walk(directory):
        file_count = len(files)
        print(f"Directory: {root}, Number of files: {file_count}")


directory_path = '/path/to/directory'
count_files_in_directory(directory_path)

