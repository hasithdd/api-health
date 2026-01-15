import polars as pl

# Add constants for feature columns and target
FEATURE_COLUMNS = [
    "is_internal_source_ip",
    "tcp_flag", "udp_flag", "icmp_flag", "ftp_flag", "http_flag", "https_flag", "ssh_flag",
    "firewall_log", "application_log", "ids_log",
    "bytes_transferred_scaled",
    "curl_flag", "windows_browser_flag", "mac_browser_flag", "nmap_script_flag", "sqlmap_flag",
    "question_mark_count", "dotdot_count", "backslash_count", "admin_keyword_count", "passwd_keyword_count", "bin_bash_keyword_count", "root_keyword_count", "hydra_keyword_count"
]
TARGET_COLUMN = "threat_label_encoded"

def load_data(path: str) -> pl.LazyFrame:
    """
    Load raw logs lazily using Polars.

    Recommended Polars APIs
    -----------------------
    - pl.scan_csv(...) for lazy, streaming-friendly loading.
    - Enables predicate pushdown and projection pushdown.

    Notes
    -----
    - Safe for multi-million-row datasets.
    - Avoid pl.read_csv unless full materialization is required.
    """
    return pl.scan_csv(
        path,
        infer_schema_length=10_000,
        ignore_errors=True
    )



def select_and_cast(df: pl.LazyFrame) -> pl.LazyFrame:
    """Select required columns and enforce correct data types.
    Columns:
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
    
        Recommended Polars APIs
    -----------------------
    - df.select([...]) to reduce column set early (projection pushdown).
    - pl.col(...).cast(...) for type enforcement.
    - Use pl.Categorical for low-cardinality string columns.

    Columns & Types
    ---------------
    - source_ip        : pl.Utf8
    - protocol         : pl.Categorical
    - threat_label     : pl.Categorical
    - log_type         : pl.Categorical
    - bytes_transferred: pl.Int64
    - user_agent       : pl.Utf8
    - request_path     : pl.Utf8

    Notes
    -----
    - Casting early improves memory usage and speed.
    - Avoid pandas-style dtype inference.
    """
    return (
        df.select([
            pl.col("source_ip").cast(pl.Utf8),
            pl.col("protocol").cast(pl.Categorical),
            pl.col("threat_label").cast(pl.Categorical),
            pl.col("log_type").cast(pl.Categorical),
            pl.col("bytes_transferred").cast(pl.Int64),
            pl.col("user_agent").cast(pl.Utf8),
            pl.col("request_path").cast(pl.Utf8),
        ])
    )    

def split_data(df: pl.DataFrame, seed: int = 42):
    """
    Split dataset into train / validation / test 70 - 10 - 20 with stratification by threat_label.
    
        Recommended Polars APIs
    -----------------------
    - df.group_by("threat_label") for stratification.
    - df.sample(frac=..., seed=...) for shuffling within groups.
    - df.slice(...) for deterministic splits.
    - pl.concat(...) to combine class-wise splits.

    Notes
    -----
    - Avoid sklearn.train_test_split to prevent pandas conversion.
    - Keeps everything in Polars memory layout.
    - Preserves class distribution within each split.
    """
    groups = df.group_by("threat_label")
    train_parts = []
    val_parts = []
    test_parts = []
    for group_key, subgroup in groups:
        subgroup = subgroup.sample(n=subgroup.height, seed=seed, shuffle=True)
        n = subgroup.height
        train_end = int(0.7 * n)
        val_end = int(0.8 * n)
        train_parts.append(subgroup.slice(0, train_end))
        val_parts.append(subgroup.slice(train_end, val_end - train_end))
        test_parts.append(subgroup.slice(val_end, n - val_end))
    train = pl.concat(train_parts)
    val = pl.concat(val_parts)
    test = pl.concat(test_parts)
    return train, val, test

def add_ip_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add internal/external IP flags for source_ip.
    Logic
    -----
    - Internal IPs start with '192.168.'.
    - All others are considered external.

    Recommended Polars APIs
    -----------------------
    - pl.col("source_ip").str.starts_with(...)
    - .cast(pl.Int8) for compact binary flags.

    Example
    -------
    internal_ip_flag = pl.col("source_ip").str.starts_with("192.168.").cast(pl.Int8)

    Notes
    -----
    - Avoid regex; prefix matching is faster.
    """
    return df.with_columns(
        pl.col("source_ip")
        .str.starts_with("192.168.")
        .cast(pl.Int8)
        .alias("is_internal_source_ip")
    )

def add_protocol_type_features(df: pl.DataFrame) -> pl.DataFrame:
    """Abstract protocol types into behavioral flags.
    flags - tcp_flag for 'TCP', udp_flag for 'UDP', icmp_flag for 'ICMP', ftp_flag for 'FTP', http_flag for 'HTTP', https_flag for 'HTTPS', ssh_flag for 'SSH'.
    
    Flags
    -----
    - tcp_flag, udp_flag, icmp_flag, ftp_flag,
      http_flag, https_flag, ssh_flag

    Recommended Polars APIs
    -----------------------
    - pl.when(pl.col("protocol") == "TCP").then(1).otherwise(0)
    - .cast(pl.Int8)

    Notes
    -----
    - Equality checks are faster than regex or contains().
    - Use Categorical dtype for protocol column.
    """
    protocols = ["TCP", "UDP", "ICMP", "FTP", "HTTP", "HTTPS", "SSH"]

    return df.with_columns([
        (pl.col("protocol") == p).cast(pl.Int8).alias(f"{p.lower()}_flag")
        for p in protocols
    ])

def add_treatment_label_encoding(df: pl.DataFrame) -> pl.DataFrame:
    """Encode threat_label into numerical labels.
    benign - 0, suspicious - 1, malicious - 2.
    
    Encoding Scheme
    ---------------
    - benign     → 0
    - suspicious → 1
    - malicious  → 2

    Recommended Polars APIs
    -----------------------
    - pl.col("threat_label").map_elements(...)
    - or pl.when(...).then(...).otherwise(...)

    Notes
    -----
    - Do NOT use sklearn.LabelEncoder.
    - Deterministic and fully lazy-compatible.
    """
    return df.with_columns(
        pl.when(pl.col("threat_label") == "benign").then(0)
        .when(pl.col("threat_label") == "suspicious").then(1)
        .otherwise(2)
        .cast(pl.Int8)
        .alias("threat_label_encoded")
    )

def add_log_type_features(df: pl.DataFrame) -> pl.DataFrame:
    """Abstract log types into behavioral flags.
    flags - firewall_log for 'firewall', application_log for 'application', ids_log for 'ids'.
    
    Flags
    -----
    - firewall_log
    - application_log
    - ids_log

    Recommended Polars APIs
    -----------------------
    - pl.col("log_type") == "firewall"
    - .cast(pl.Int8)

    Notes
    -----
    - String equality is faster than substring matching.
    """
    return df.with_columns([
        (pl.col("log_type") == "firewall").cast(pl.Int8).alias("firewall_log"),
        (pl.col("log_type") == "application").cast(pl.Int8).alias("application_log"),
        (pl.col("log_type") == "ids").cast(pl.Int8).alias("ids_log"),
    ])

def add_bytes_transferred_scaling(df: pl.DataFrame, mean: float, std: float) -> pl.DataFrame:
    """
    Standardize bytes_transferred using z-score normalization.

    Formula
    -------
    (x - mean) / std

    Recommended Polars APIs
    -----------------------
    - pl.col("bytes_transferred").mean()
    - pl.col("bytes_transferred").std()
    - Arithmetic expressions inside with_columns()

    Notes
    -----
    - Prefer Polars over sklearn.StandardScaler.
    - Fully lazy and avoids NumPy materialization.
    """
    return df.with_columns(
        ((pl.col("bytes_transferred") - mean) / std)
        .alias("bytes_transferred_scaled")
    )

def add_user_agent_features(df: pl.DataFrame) -> pl.DataFrame:
    """Abstract user agent strings into behavioral flags.
    flags - curl for curl keyword in the user agent,
      windows_browser for Windows keyword in the user agent,
      mac_browser for Macintosh keyword in the user agent,
      nmap_script for Nmap keyword in the user agent,
      SQLMap for SQLMap keyword in the user agent.

    Flags
    -----
    - curl_flag
    - windows_browser_flag
    - mac_browser_flag
    - nmap_script_flag
    - sqlmap_flag

    Recommended Polars APIs
    -----------------------
    - pl.col("user_agent").str.contains("curl", literal=True)
    - Use lowercase normalization via .str.to_lowercase()

    Notes
    -----
    - Avoid regex unless absolutely necessary.
    - literal=True is faster and safer.
    """
    ua = pl.col("user_agent").str.to_lowercase()

    return df.with_columns([
        ua.str.contains("curl", literal=True).cast(pl.Int8).alias("curl_flag"),
        ua.str.contains("windows", literal=True).cast(pl.Int8).alias("windows_browser_flag"),
        ua.str.contains("macintosh", literal=True).cast(pl.Int8).alias("mac_browser_flag"),
        ua.str.contains("nmap", literal=True).cast(pl.Int8).alias("nmap_script_flag"),
        ua.str.contains("sqlmap", literal=True).cast(pl.Int8).alias("sqlmap_flag"),
    ])

def add_path_structure_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract structural features from request_path.
    Features & Methods
    ------------------
    - question_mark_count       → str.count_matches("?")
    - SQL_command_count         → sum of str.count_matches(...) for keywords
    - backslash_count           → str.count_matches("\\\\")
    - admin_keyword_count       → str.count_matches("admin")
    - passwd_keyword_count      → str.count_matches("passwd")
    - bin/bash_keyword_count    → str.count_matches("bin/bash")
    - root_keyword_count        → str.count_matches("root")
    - hydra_keyword_count       → str.count_matches("hydra")
    - dotdot_count              → str.count_matches("..")

    Recommended Polars APIs
    -----------------------
    - pl.col("request_path").str.to_lowercase()
    - .str.count_matches(pattern, literal=True)

    Notes
    -----
    - count_matches is vectorized and very fast.
    - Avoid Python loops at all costs.
    """
    rp = pl.col("request_path").str.to_lowercase()

    return df.with_columns([
        rp.str.count_matches("?", literal=True).alias("question_mark_count"),
        rp.str.count_matches("..", literal=True).alias("dotdot_count"),
        rp.str.count_matches("\\\\", literal=True).alias("backslash_count"),
        rp.str.count_matches("admin", literal=True).alias("admin_keyword_count"),
        rp.str.count_matches("passwd", literal=True).alias("passwd_keyword_count"),
        rp.str.count_matches("bin/bash", literal=True).alias("bin_bash_keyword_count"),
        rp.str.count_matches("root", literal=True).alias("root_keyword_count"),
        rp.str.count_matches("hydra", literal=True).alias("hydra_keyword_count"),
    ])

def save_artifacts(artifacts: dict, output_dir: str = "artifacts"):
    """Persist preprocessing artifacts."""
    import os, pickle
    os.makedirs(output_dir, exist_ok=True)

    for name, obj in artifacts.items():
        with open(f"{output_dir}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

def apply_features(df: pl.DataFrame, mean: float, std: float) -> pl.DataFrame:
    df = add_ip_flags(df)
    df = add_protocol_type_features(df)
    df = add_treatment_label_encoding(df)
    df = add_log_type_features(df)
    df = add_bytes_transferred_scaling(df, mean, std)
    df = add_user_agent_features(df)
    df = add_path_structure_features(df)
    return df.select(FEATURE_COLUMNS + [TARGET_COLUMN])



# MAIN PIPELINE (test run)

# if __name__ == "__main__":
#     file_path = "data/cybersecurity_threat_detection_logs.csv"
    
#     # Load and select/cast
#     df_lazy = load_data(file_path)
#     df_lazy = select_and_cast(df_lazy)
    
#     # Lightweight schema validation
#     required_cols = ["source_ip", "protocol", "threat_label", "log_type", "bytes_transferred", "user_agent", "request_path"]
#     schema_names = df_lazy.collect_schema().names()
#     for col in required_cols:
#         assert col in schema_names, f"Missing required column: {col}"
#     assert df_lazy.select(pl.col("threat_label").is_null().sum()).collect().item() == 0, "threat_label contains null values"
    
#     # Collect to DataFrame for splitting
#     df = df_lazy.collect()
    
#     # Stratified split
#     train, val, test = split_data(df)
    
#     # Compute scaling stats from train only
#     mean = train.select(pl.col("bytes_transferred").mean()).item()
#     std = train.select(pl.col("bytes_transferred").std()).item()
    
#     # Apply feature engineering to each split
#     train = apply_features(train, mean, std)
#     val = apply_features(val, mean, std)
#     test = apply_features(test, mean, std)
    
#     # Save artifacts including metadata
#     artifacts = {
#         "train": train,
#         "val": val,
#         "test": test,
#         "feature_columns": FEATURE_COLUMNS,
#         "bytes_transferred_mean": mean,
#         "bytes_transferred_std": std,
#         "label_mapping": {"benign": 0, "suspicious": 1, "malicious": 2}
#     }
#     save_artifacts(artifacts)

def main():
    file_path = "data/cybersecurity_threat_detection_logs.csv"

    df_lazy = load_data(file_path)
    df_lazy = select_and_cast(df_lazy)

    required_cols = [
        "source_ip", "protocol", "threat_label",
        "log_type", "bytes_transferred", "user_agent", "request_path"
    ]
    schema_names = df_lazy.collect_schema().names()
    for col in required_cols:
        assert col in schema_names, f"Missing required column: {col}"

    df = df_lazy.collect()

    train, val, test = split_data(df)

    mean = train.select(pl.col("bytes_transferred").mean()).item()
    std = train.select(pl.col("bytes_transferred").std()).item()

    train = apply_features(train, mean, std)
    val = apply_features(val, mean, std)
    test = apply_features(test, mean, std)

    artifacts = {
        "train": train,
        "val": val,
        "test": test,
        "feature_columns": FEATURE_COLUMNS,
        "bytes_transferred_mean": mean,
        "bytes_transferred_std": std,
        "label_mapping": {"benign": 0, "suspicious": 1, "malicious": 2}
    }
    save_artifacts(artifacts)
