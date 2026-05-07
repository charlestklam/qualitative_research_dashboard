import json
import pandas as pd

class JSONProcessor:
    """Class for processing JSON files and converting them to analyzable formats."""
    
    @staticmethod
    def parse_json(json_string):
        """
        Parse a JSON string into a Python object
        
        Args:
            json_string (str): JSON data as a string
            
        Returns:
            dict/list: Parsed JSON data
            
        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    @staticmethod
    def flatten_json(data, prefix=""):
        """
        Recursively flatten a nested JSON structure into a flat dictionary
        
        Args:
            data (dict/list): JSON data to flatten
            prefix (str): Prefix for nested keys
            
        Returns:
            dict: Flattened dictionary with dot notation for nested keys
        """
        items = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    items.update(JSONProcessor.flatten_json(value, new_key))
                else:
                    items[new_key] = value
                    
        elif isinstance(data, list):
            for i, value in enumerate(data):
                new_key = f"{prefix}[{i}]"
                
                if isinstance(value, (dict, list)):
                    items.update(JSONProcessor.flatten_json(value, new_key))
                else:
                    items[new_key] = value
        else:
            items[prefix] = data
            
        return items
    
    @staticmethod
    def json_to_df(data):
        """
        Convert JSON data to a pandas DataFrame
        
        Args:
            data (dict/list): JSON data
            
        Returns:
            pd.DataFrame: DataFrame representation of the JSON
        """
        # If data is a list of dictionaries, convert directly to DataFrame
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)
        
        # If data is a dictionary of lists with equal lengths, convert directly
        if isinstance(data, dict) and all(isinstance(item, list) for item in data.values()):
            if len(set(len(v) for v in data.values())) == 1:  # All lists have same length
                return pd.DataFrame(data)
        
        # Otherwise, flatten the JSON and convert to DataFrame
        flat_data = JSONProcessor.flatten_json(data)
        return pd.DataFrame([flat_data])
    
    @staticmethod
    def search_json(data, keyword, case_sensitive=False, whole_word=False):
        """
        Search for a keyword in a JSON structure
        
        Args:
            data (dict/list): JSON data to search through
            keyword (str): Keyword to search for
            case_sensitive (bool): Whether to perform case-sensitive search
            whole_word (bool): Whether to match whole words only
            
        Returns:
            list: List of paths where the keyword was found
        """
        import re
        results = []
        flat_data = JSONProcessor.flatten_json(data)
        
        # Define regex for whole word search 
        regex = None
        if whole_word:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if case_sensitive:
                regex = re.compile(pattern)
            else:
                regex = re.compile(pattern, re.IGNORECASE)
        
        for path, value in flat_data.items():
            if isinstance(value, str):
                if whole_word and regex:
                    # Use regex for whole word matching
                    if regex.search(value):
                        results.append((path, value))
                else:
                    # Use standard substring search
                    if case_sensitive:
                        if keyword in value:
                            results.append((path, value))
                    else:
                        if keyword.lower() in value.lower():
                            results.append((path, value))
            elif isinstance(value, (int, float, bool)):
                # Convert to string and check
                str_value = str(value)
                
                if whole_word and regex:
                    if regex.search(str_value):
                        results.append((path, str_value))
                else:
                    if case_sensitive:
                        if keyword in str_value:
                            results.append((path, str_value))
                    else:
                        if keyword.lower() in str_value.lower():
                            results.append((path, str_value))
                    
        return results
    
    @staticmethod
    def extract_text_fields(data):
        """
        Extract all text fields from a JSON structure
        
        Args:
            data (dict/list): JSON data
            
        Returns:
            list: List of all text values
        """
        text_values = []
        flat_data = JSONProcessor.flatten_json(data)
        
        for _, value in flat_data.items():
            if isinstance(value, str) and len(value.split()) > 1:  # Consider only values with multiple words
                text_values.append(value)
                
        return text_values
