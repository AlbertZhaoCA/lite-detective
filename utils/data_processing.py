import csv
import json
import os
from typing import List

def clean_data_bom(data:str):
    """
    Remove BOM characters from the beginning of a string.
    
    Args:
        data (str): The input string potentially containing BOM characters.
        
    Returns:
        str: The cleaned string without BOM characters.
    """
    return data.lstrip('\ufeff')
    
def clean_data(data:str):
    """
    Wrapper function to clean data by removing BOM characters and leading/trailing whitespace.
    
    Args:
        data (str): The input string to be cleaned.
        
    Returns:
        str: The cleaned string.
    """
    data = clean_data_bom(data)
    
    return data.strip()

def remove_json_decorators(data: str) -> str:
    """
    Remove JSON decorators from a string, including '''json and ''' marks.
    
    Args:
        data (str): The input string containing JSON decorators.
        
    Returns:
        str: The cleaned string without JSON decorators.
    """
    if data.startswith("'''json"):
        data = data[len("'''json"):].strip()  # Remove leading '''json
    if data.endswith("'''"):
        data = data[:-3].strip()  # Remove trailing '''
    
    return data

def get_all_files(dir_path: str, file_extension: str="json") -> List[str]:
    """
    Get all files in a directory with a specific file extension.
    
    Args:
        dir_path (str): The path to the directory.
        file_extension (str): The file extension to search for.
        
    Returns:
        List[str]: A list of file paths.
    """
    files = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(file_extension):
                files.append(os.path.join(root, filename))
    return files

def load_csv(file_path) -> list:
    """
    Load a CSV file and return a list 
    """   
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = [list(row.values()) for row in reader]
    return data

def save_list_to_csv(data, file_path):
    """
    Save a list of lists to a CSV file.

    Args:
        data (list of lists): The data to save.
            the first row is the header.
            [["column1", "column2"], ["value1", "value2"],["value3", "value4"],...]
        file_path (str): The path to the CSV file.
    """
    dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist. Creating it.")
        os.makedirs(os.path.dirname(file_path))
    if not data:
        return
    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data[0])
        for row in data[1:]:
            writer.writerow(row)
    print(f"Data saved to {file_path}")

def write_json_to_file(file_path, journal):
    """
    Write a json object to a json file.
    """
    dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(dir):
        print(f"Directory {dir} does not exist. Creating it.")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
    with open(file_path, mode='w', newline='', encoding='utf-8') as journal_file:
       journal = json.dumps(journal, indent=4)
       journal_file.write(journal)

def read_json_from_file(file_path):
    """
    Read a json obj from a json file.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    with open(file_path, mode='r', encoding='utf-8') as journal_file:
        journal = json.load(journal_file)
    return journal



if __name__ == "__main__":
    # Example usage
    data = "'''json\n{\"key\": \"value\"}'''"
    cleaned_data = remove_json_decorators(data)
    print(cleaned_data)  # Output: {"key": "value"}
    
    data_with_bom = "\ufeffHello, World!"
    cleaned_data = clean_data(data_with_bom)
    print(cleaned_data)  # Output: Hello, World!