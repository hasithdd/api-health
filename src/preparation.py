import polars as pl
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder



def load_data(path: str) -> pl.LazyFrame:
    """Load raw logs lazily using Polars."""
    return pl.scan_csv(path)



def select_and_cast(df: pl.LazyFrame) -> pl.LazyFrame:
    """Select required columns and enforce correct data types.
    # Columns:
    timestamp: not required
    source_ip: str (later to be converted to categorical as internal / external)
    dest_ip: str (not required)
    protocol: categorical [str] (later to be encoded as protocol types)
    action: not required
    threat_label: categorical [str] (target variable to be label encoded to benign, suspicious, malicious)
    log_type: categorical [str] (later to be encoded as log types - firewall, application, ids)
    bytes_transferred: numeric [int] (later to be scaled [standard scaling])
    user_agent: str (later to be converted to categorical behavioral flags - curl, windows_browser, mac_browser, nmap_script, SQLMap.)
    request_path: str (later to be converted to structural features - "question_mark_count", "SQL_command_count", "backslash_count", "admin_keyword_count", "passwd_keyword_count", "bin/bash_keyword_count")
    """

def split_data(df: pl.DataFrame, seed: int = 42):
    """
    Split dataset into train / validation / test 70 - 10 - 20 .
    """


def add_ip_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add internal/external IP flags for source_ip.
    ips with starting with '192.168.' internal, else external.
    """

def add_protocol_type_features(df: pl.DataFrame) -> pl.DataFrame:
    """Abstract protocol types into behavioral flags.
    flags - tcp_flag for 'TCP', udp_flag for 'UDP', icmp_flag for 'ICMP', ftp_flag for 'FTP', http_flag for 'HTTP', https_flag for 'HTTPS', ssh_flag for 'SSH'.
    """

def add_treatment_label_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """Encode threat_label into numerical labels.
    benign - 0, suspicious - 1, malicious - 2.
    """

def add_log_type_features(df: pl.DataFrame) -> pl.DataFrame:
    """Abstract log types into behavioral flags.
    flags - firewall_log for 'firewall', application_log for 'application', ids_log for 'ids'.
    """

def add_bytes_transferred_scaling(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a standardized (z-score) version of the `bytes_transferred` column.

    The feature is scaled using mean and standard deviation computed from the
    input DataFrame, producing a zero-mean, unit-variance representation suitable
    for magnitude-sensitive machine learning models.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the `bytes_transferred` column.

    Returns
    -------
    pl.DataFrame
        DataFrame with an additional scaled `bytes_transferred` feature.
    """

def add_user_agent_features(df: pl.DataFrame) -> pl.DataFrame:
    """Abstract user agent strings into behavioral flags.
    flags - curl for curl keyword in the user agent,
      windows_browser for Windows keyword in the user agent,
      mac_browser for Macintosh keyword in the user agent,
      nmap_script for Nmap keyword in the user agent,
      SQLMap for SQLMap keyword in the user agent.
    """

def add_path_structure_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract structural features from request_path.
    features - "question_mark_count" for count of '?' characters,
     "SQL_command_count" for count of SQL commands (SELECT, INSERT, UPDATE, DELETE),
     "backslash_count" for count of '\' characters,
     "admin_keyword_count" for count of 'admin' keyword occurrences,
     "passwd_keyword_count" for count of 'passwd' keyword occurrences,
     "bin/bash_keyword_count" for count of 'bin/bash' keyword occurrences,
     "root_keyword_count" for count of 'root' keyword occurrences.
     "hydra_keyword_count" for count of 'hydra' keyword occurrences.
     ".._count" for count of '..' sequences.
    """


def encode_and_scale(train_df, val_df, test_df):
    """Encode categorical features, scale numerics, encode target."""

def save_artifacts(artifacts: dict, output_dir: str = "artifacts"):
    """Persist preprocessing artifacts."""



# MAIN PIPELINE (test run)

if __name__ == "__main__":
    file_path = "data/cybersecurity_threat_detection_logs.csv"

