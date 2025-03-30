import pandas as pd
from typing import List

class TranscriptReader:
    def __init__(self, file_path: str):
        """
        Initialize the transcript reader with a file path.
        
        Args:
            file_path (str): Path to the CSV file containing transcripts
        """
        self.file_path = file_path
        self.df = None
        
    def load_data(self) -> None:
        """Load the CSV data into a pandas DataFrame."""
        self.df = pd.read_csv(self.file_path)
        
    def get_messages(self) -> List[str]:
        """
        Get all messages from the transcript.
        
        Returns:
            List[str]: List of messages from the transcript
        """
        if self.df is None:
            self.load_data()
        return self.df['Message'].tolist()
    
    def get_message_at_index(self, index: int) -> str:
        """
        Get a specific message at the given index.
        
        Args:
            index (int): Index of the message to retrieve
            
        Returns:
            str: Message at the specified index
        """
        if self.df is None:
            self.load_data()
        return self.df.iloc[index]['Message']
    

# Example usage
reader = TranscriptReader("../data/transcripts/c2.csv")

# Get all messages
messages = reader.get_messages()

# Get a specific message at index 0
first_message = reader.get_message_at_index(0)
print(first_message)