import pandas as pd
from typing import List
import re

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
        Get all user messages from the transcript.
        
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
    
    def get_all_messages_string(self) -> str:
        """
        Get all messages concatenated into a single string.
        
        Returns:
            str: All messages concatenated with newlines between them
        """
        if self.df is None:
            self.load_data()
        return '\n'.join(self.df['Message'].tolist())
    
    def _clean_npc_message(self, message: str) -> str:
        """
        Clean NPC message by removing the emotion/action part in parentheses.
        
        Args:
            message (str): Raw NPC message
            
        Returns:
            str: Cleaned message without the emotion/action part
        """
        # Remove the part in parentheses at the start of the message
        cleaned = re.sub(r'^\([^)]*\)\s*', '', message)
        return cleaned.strip()
    
    def get_npc_messages(self) -> List[str]:
        """
        Get all NPC messages from the transcript.
        
        Returns:
            List[str]: List of cleaned NPC messages
        """
        if self.df is None:
            self.load_data()
        return [self._clean_npc_message(msg) for msg in self.df['NPC Response'].tolist()]
    
    def get_all_npc_messages_string(self) -> str:
        """
        Get all NPC messages concatenated into a single string.
        
        Returns:
            str: All cleaned NPC messages concatenated with newlines between them
        """
        if self.df is None:
            self.load_data()
        return '\n'.join(self.get_npc_messages())
    

# Example usage
reader = TranscriptReader("../data/transcripts/c2.csv")

# Get all messages
messages = reader.get_messages()
print(messages)

# Get all messages as a single string
all_messages = reader.get_all_messages_string()
print(all_messages)

# Get all NPC messages
npc_messages = reader.get_npc_messages()
print("\nNPC Messages:")
print(npc_messages)

# Get all NPC messages as a single string
all_npc_messages = reader.get_all_npc_messages_string()
print("\nAll NPC Messages as string:")
print(all_npc_messages)