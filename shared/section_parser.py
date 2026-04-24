import pandas as pd 
import re

def parse_sections(text):
    """
    Parse a clinical note into a dictionary of labelled sections.

    Uses a regex pattern to identify section headers in ALL CAPS with
    optional trailing colon. Handles SOAP and non-SOAP templates including
    headers with '/', '&' separators. Content before the first recognised
    header is stored under 'UNKNOWN'.

    Args:
        text (str): Raw transcription text.

    Returns:
        dict[str, str]: Dictionary mapping section label to section content.
    """
    if not isinstance(text, str):
        return {}
    sections = re.split(r'^([A-Z][A-Z\s/&]+):?\s*$', text, flags=re.MULTILINE)
    if not sections[0].strip():
        sections = sections[1:]

    it = iter(sections)
    sections_tuple = list(zip(it,it))
    sections_dict = dict(sections_tuple)
    return sections_dict 

def parse_sections_df(df: pd.DataFrame):
    
    """
    Apply section parser to all transcription records in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'transcription' column.

    Returns:
        pd.DataFrame: Input DataFrame with an additional 'sections' column
            containing a dict of parsed section labels to content per record.
    """

    
    df["sections"] = df["transcription"].apply(parse_sections)
    return df