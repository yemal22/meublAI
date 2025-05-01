import os

def get_path(output_path:str) -> str:
    """
    Generates a checkpoint file path by appending the '.checkpoint' extension 
    to the provided output path.

    Args:
        output_path (str): The base path where the checkpoint file will be saved.

    Returns:
        str: The complete file path with the '.checkpoint' extension.
    """
    return f"{output_path}.checkpoint"

def load(output_path:str) -> int:
    """
    Load the checkpoint value from a file.

    This function reads an integer value from a checkpoint file located at the
    specified output path. If the file exists, it reads and returns the integer
    value stored in the file. If the file does not exist, it returns 0.

    Args:
        output_path (str): The directory path where the checkpoint file is located.

    Returns:
        int: The integer value read from the checkpoint file, or 0 if the file does not exist.
    """
    checkpoint_file = get_path(output_path)
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save(output_path:str, index:int) -> None:
    """
    Saves the given index to a checkpoint file at the specified output path.

    Args:
        output_path (str): The directory path where the checkpoint file will be saved.
        index (int): The index value to be written to the checkpoint file.

    The function creates or overwrites a file in the specified directory and writes
    the index value as a string to the file.
    """
    checkpoint_file = get_path(output_path)
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))