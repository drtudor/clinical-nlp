import re

import pandas as pd
import pytest

from data_cleaning import (
    drop_boilerplate_text,
    drop_incomplete_records,
    recover_nan_keywords,
    remove_specialty,
    parse_sections,
    parse_sections_df,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df():
    """Minimal valid MTSamples-style dataframe for use across tests."""
    return pd.DataFrame({
        "keywords": [
            "chest pain, dyspnea, hypertension",
            "fracture, orthopaedic surgery",
            None,
            "diabetes mellitus, insulin resistance",
            None,
        ],
        "transcription": [
            "HISTORY: Patient presents with chest pain.\nASSESSMENT: Hypertension, dyspnea.",
            "SUBJECTIVE: Fell from height.\nASSESSMENT: Fracture of left femur.",
            "HISTORY: Cough and fever.\nDIAGNOSIS: Upper respiratory tract infection.",
            "ASSESSMENT: Diabetes mellitus type 2.",
            "Short note.",
        ],
        "transcription_length": [500, 600, 550, 510, 80],
        "medical_specialty": [
            " Cardiovascular / Pulmonary",
            " Orthopedic",
            " General Medicine",
            " Endocrinology",
            " General Medicine",
        ],
    })


# ---------------------------------------------------------------------------
# drop_incomplete_records
# ---------------------------------------------------------------------------

class TestDropIncompleteRecords:

    def test_drops_nan_keywords_with_short_transcription(self, base_df):
        """Rows with NaN keywords AND short transcription should be removed."""
        result = drop_incomplete_records(base_df)
        assert len(result) == 4

    def test_retains_nan_keywords_with_long_transcription(self, base_df):
        """Rows with NaN keywords but long transcription should be retained."""
        result = drop_incomplete_records(base_df)
        nan_rows = result[result["keywords"].isna()]
        assert len(nan_rows) == 1
        assert nan_rows.iloc[0]["transcription_length"] == 550

    def test_retains_valid_records(self, base_df):
        """Records with keywords present should never be dropped."""
        result = drop_incomplete_records(base_df)
        non_nan = result[result["keywords"].notna()]
        assert len(non_nan) == 3

    def test_returns_dataframe(self, base_df):
        assert isinstance(drop_incomplete_records(base_df), pd.DataFrame)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"keywords": [], "transcription_length": []})
        result = drop_incomplete_records(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# drop_boilerplate_text
# ---------------------------------------------------------------------------

class TestDropBoilerplateText:

    def test_removes_long_keywords(self):
        """Keywords exceeding 50 characters should be removed."""
        df = pd.DataFrame({
            "keywords": [
                "chest pain, this is a very long boilerplate disclaimer text that should be removed, dyspnea"
            ]
        })
        result = drop_boilerplate_text(df)
        keywords = result["keywords"].dropna().tolist()
        assert not any(len(k) >= 50 for k in keywords)

    def test_retains_short_keywords(self):
        """Keywords under 50 characters should be retained."""
        df = pd.DataFrame({"keywords": ["chest pain, dyspnea, hypertension"]})
        result = drop_boilerplate_text(df)
        assert result["keywords"].notna().any()

    def test_handles_nan_keywords(self):
        """NaN keywords should not raise an error."""
        df = pd.DataFrame({"keywords": [None, "chest pain"]})
        result = drop_boilerplate_text(df)
        assert isinstance(result, pd.DataFrame)

    def test_strips_whitespace(self):
        """Keywords should have leading/trailing whitespace removed."""
        df = pd.DataFrame({"keywords": ["  chest pain  ,  dyspnea  "]})
        result = drop_boilerplate_text(df)
        keywords = result["keywords"].dropna().tolist()
        assert all(k == k.strip() for k in keywords)

    def test_returns_dataframe(self):
        df = pd.DataFrame({"keywords": ["chest pain, dyspnea"]})
        assert isinstance(drop_boilerplate_text(df), pd.DataFrame)


# ---------------------------------------------------------------------------
# remove_specialty
# ---------------------------------------------------------------------------

class TestRemoveSpecialty:

    def test_removes_specialty_labels(self, base_df):
        """Specialty labels should not appear in cleaned keywords."""
        result = remove_specialty(base_df)
        specialty_terms = {"cardiovascular", "pulmonary", "orthopedic",
                           "general medicine", "endocrinology"}
        keywords = result["keywords"].dropna().tolist()
        assert not any(k.lower() in specialty_terms for k in keywords)

    def test_retains_clinical_terms(self, base_df):
        """Clinical terms should be preserved after specialty removal."""
        result = remove_specialty(base_df)
        keywords = result["keywords"].dropna().tolist()
        assert any("chest pain" in k or "dyspnea" in k for k in keywords)

    def test_handles_compound_specialties(self):
        """Specialties with '/' should be split and both terms filtered."""
        df = pd.DataFrame({
            "keywords": ["cardiovascular, chest pain, pulmonary"],
            "transcription_length": [500],
            "transcription": [""],
            "medical_specialty": [" Cardiovascular / Pulmonary"],
        })
        result = remove_specialty(df)
        keywords = result["keywords"].dropna().tolist()
        assert not any(k in {"cardiovascular", "pulmonary"} for k in keywords)

    def test_returns_dataframe(self, base_df):
        assert isinstance(remove_specialty(base_df), pd.DataFrame)


# ---------------------------------------------------------------------------
# recover_nan_keywords
# ---------------------------------------------------------------------------

class TestRecoverNanKeywords:

    def test_recovers_from_assessment_section(self):
        """Terms from ASSESSMENT section should be recovered into keywords."""
        df = pd.DataFrame({
            "keywords": [None],
            "transcription": ["HISTORY: Patient unwell.\nASSESSMENT: Hypertension, type 2 diabetes."],
            "transcription_length": [550],
            "medical_specialty": [" General Medicine"],
        })
        result = recover_nan_keywords(df)
        assert result.iloc[0]["keywords"] is not None
        assert "Hypertension" in result.iloc[0]["keywords"]

    def test_recovers_from_diagnosis_section(self):
        """Terms from DIAGNOSIS section should be recovered into keywords."""
        df = pd.DataFrame({
            "keywords": [None],
            "transcription": ["SUBJECTIVE: Cough.\nDIAGNOSIS: Upper respiratory infection."],
            "transcription_length": [550],
            "medical_specialty": [" General Medicine"],
        })
        result = recover_nan_keywords(df)
        assert result.iloc[0]["keywords"] is not None

    def test_does_not_overwrite_existing_keywords(self, base_df):
        """Existing non-NaN keywords should not be modified."""
        original = base_df.iloc[0]["keywords"]
        result = recover_nan_keywords(base_df)
        assert result.iloc[0]["keywords"] == original

    def test_returns_none_for_no_target_section(self):
        """Records with no ASSESSMENT/DIAGNOSIS section should remain NaN."""
        df = pd.DataFrame({
            "keywords": [None],
            "transcription": ["HISTORY: Patient presented with cough."],
            "transcription_length": [550],
            "medical_specialty": [" General Medicine"],
        })
        result = recover_nan_keywords(df)
        assert result.iloc[0]["keywords"] is None

    def test_handles_non_string_transcription(self):
        """Non-string transcription values should not raise an error."""
        df = pd.DataFrame({
            "keywords": [None],
            "transcription": [None],
            "transcription_length": [550],
            "medical_specialty": [" General Medicine"],
        })
        result = recover_nan_keywords(df)
        assert isinstance(result, pd.DataFrame)

    def test_returns_dataframe(self, base_df):
        assert isinstance(recover_nan_keywords(base_df), pd.DataFrame)

# ---------------------------------------------------------------------------
# parse_sections
# ---------------------------------------------------------------------------

class TestParseSections:

    def test_parses_standard_soap_note(self):
        """Standard SOAP headers should be correctly parsed into sections."""
        text = "SUBJECTIVE:\nPatient has chest pain.\nASSESSMENT:\nHypertension."
        result = parse_sections(text)
        assert "SUBJECTIVE:" in result or "SUBJECTIVE" in result
        assert "chest pain" in result.get("SUBJECTIVE:", result.get("SUBJECTIVE", ""))

    def test_parses_header_without_colon(self):
        """Headers without trailing colon should still be detected."""
        text = "ASSESSMENT\nHypertension."
        result = parse_sections(text)
        assert any("ASSESSMENT" in k for k in result.keys())

    def test_parses_slash_header(self):
        """Headers with '/' such as ASSESSMENT/PLAN should be detected."""
        text = "ASSESSMENT/PLAN:\nHypertension. Follow up in 2 weeks."
        result = parse_sections(text)
        assert any("ASSESSMENT/PLAN" in k for k in result.keys())

    def test_returns_empty_dict_for_none(self):
        """None input should return an empty dict."""
        assert parse_sections(None) == {}

    def test_returns_empty_dict_for_non_string(self):
        """Non-string input should return an empty dict."""
        assert parse_sections(123) == {}

    def test_content_before_first_header_excluded(self):
        """Content before the first header should not appear in section values."""
        text = "Some preamble text.\nSUBJECTIVE:\nPatient presents well."
        result = parse_sections(text)
        assert "preamble" not in str(result.values())


# ---------------------------------------------------------------------------
# parse_sections_df
# ---------------------------------------------------------------------------

class TestParseSectionsDf:

    def test_adds_sections_column(self, base_df):
        """Output DataFrame should contain a 'sections' column."""
        result = parse_sections_df(base_df)
        assert "sections" in result.columns

    def test_sections_column_contains_dicts(self, base_df):
        """Every value in the sections column should be a dict."""
        result = parse_sections_df(base_df)
        assert all(isinstance(s, dict) for s in result["sections"])

    def test_returns_dataframe(self, base_df):
        assert isinstance(parse_sections_df(base_df), pd.DataFrame)

    def test_row_count_unchanged(self, base_df):
        """Row count should be unchanged after parsing."""
        result = parse_sections_df(base_df)
        assert len(result) == len(base_df)

