import re
import sqlite3

import pandas as pd


def drop_incomplete_records(df):
    """
    Remove records where keywords are missing and transcription is short.

    Drops rows where keywords is NaN AND transcription_length < 500,
    as these records lack sufficient signal for downstream NLP tasks.

    Args:
        df (pd.DataFrame): Raw MTSamples dataframe.

    Returns:
        pd.DataFrame: Dataframe with incomplete records removed.
    """
    df = df[~((df['keywords'].isna()) & (df['transcription_length'] < 500))]
    return df


def drop_boilerplate_text(df):
    """
    Strip boilerplate disclaimer text from the keywords column.

    Splits keywords on commas, removes whitespace and empty strings,
    and filters out entries exceeding 50 characters, which empirically
    correspond to boilerplate rather than clinical terms.

    Args:
        df (pd.DataFrame): Dataframe with a 'keywords' column.

    Returns:
        pd.DataFrame: Dataframe with cleaned keywords column.
    """
    keywords = df['keywords'].str.split(',').explode().str.strip()
    keywords = keywords[keywords != '']
    keywords_clean = keywords[keywords.str.len() < 50]
    df['keywords'] = keywords_clean.groupby(level=0).agg(', '.join)
    return df


def remove_specialty(df):
    """
    Remove medical specialty labels from the keywords column.

    Builds a list of known specialty terms from the medical_specialty column,
    splits on '/' to handle compound entries, and filters out any keywords
    that match a specialty label. Only clinical terms are retained.

    Args:
        df (pd.DataFrame): Dataframe with 'keywords' and 'medical_specialty' columns.

    Returns:
        pd.DataFrame: Dataframe with specialty labels removed from keywords.
    """
    specialty_keywords = (pd.Series(df['medical_specialty'].unique())
                          .str.lower()
                          .str.split('/')
                          .explode()
                          .str.strip())

    keywords = (df['keywords']
                .str.split(',')
                .explode()
                .str.strip())
    keywords = keywords[keywords != '']

    keywords_filtered = keywords[~keywords.isin(specialty_keywords)].dropna()
    df['keywords'] = keywords_filtered.groupby(level=0).agg(', '.join)
    return df


def recover_nan_keywords(df):
    """
    Recover keywords for records where the keywords column is NaN.

    Identifies rows with missing keywords, extracts text from ASSESSMENT
    or DIAGNOSIS sections using a flexible regex section parser, and assigns
    the extracted terms back into the keywords column.

    Args:
        df (pd.DataFrame): Dataframe with 'keywords' and 'transcription' columns.

    Returns:
        pd.DataFrame: Dataframe with NaN keywords partially recovered.
    """
    target_sections = {'ASSESSMENT', 'DIAGNOSIS', 'IMPRESSION'}

    def extract_section_terms(text):
        if not isinstance(text, str):
            return None
        sections = re.split(r'(^[A-Z][A-Z\s]+:?)', text, flags=re.MULTILINE)
        terms = []
        capture = False
        for chunk in sections:
            header = chunk.strip().rstrip(':')
            if header in target_sections:
                capture = True
            elif re.match(r'^[A-Z][A-Z\s]+:?$', chunk.strip()):
                capture = False
            elif capture:
                terms.extend([t.strip() for t in chunk.split(',') if t.strip()])
        return ', '.join(terms) if terms else None

    mask = df['keywords'].isna()
    df.loc[mask, 'keywords'] = df.loc[mask, 'transcription'].apply(extract_section_terms)
    return df


def main():
    """
    Execute the full data cleaning pipeline and save output to SQLite.

    Loads raw MTSamples data, applies cleaning steps in sequence, and
    saves the cleaned dataset to a SQLite database for downstream use.

    Steps:
        1. Drop incomplete records (NaN keywords + short transcriptions)
        2. Strip boilerplate text from keywords
        3. Remove specialty labels from keywords
        4. Recover keywords for NaN records from ASSESSMENT/DIAGNOSIS sections

    Output:
        Writes cleaned data to ../data/processed/mtsamples_cleaned.db
    """
    df = pd.read_csv("../data/raw/mtsamples.csv", index_col=0)
    df = drop_incomplete_records(df)
    df = drop_boilerplate_text(df)
    df = remove_specialty(df)
    df = recover_nan_keywords(df)

    conn = sqlite3.connect("../data/processed/mtsamples_cleaned.db")
    df.to_sql("mtsamples", conn, if_exists="replace", index=False)
    conn.close()

    print("Cleaned dataset saved to ../data/processed/mtsamples_cleaned.db")


if __name__ == "__main__":
    main()