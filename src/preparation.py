import polars as pl
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder



def load_data(path: str) -> pl.LazyFrame:
    """Load raw logs lazily using Polars."""
    return pl.scan_csv(path)



def select_and_cast(df: pl.LazyFrame) -> pl.LazyFrame:
    """Select required columns and enforce correct data types."""
    return (
        df.select([
            "source_ip",
            "destination_ip",
            "request_path",
            "protocol",
            "action",
            "log_type",
            "user_agent",
            "bytes_transferred",
            "threat_label",
        ])
        .with_columns([
            pl.col("bytes_transferred").cast(pl.Int32),
            pl.col("threat_label").cast(pl.Utf8),
            pl.col("protocol").cast(pl.Utf8),
            pl.col("action").cast(pl.Utf8),
            pl.col("log_type").cast(pl.Utf8),
            pl.col("user_agent").cast(pl.Utf8),
            pl.col("request_path").cast(pl.Utf8),
        ])
    )



def split_data(df: pl.DataFrame, seed: int = 42):
    """Split dataset into train / validation / test (70/15/15)."""
    y = df["threat_label"]

    splitter_1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.3, random_state=seed
    )
    train_idx, temp_idx = next(splitter_1.split(df, y))

    train_df = df[train_idx]
    temp_df = df[temp_idx]

    splitter_2 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.5, random_state=seed
    )
    val_idx, test_idx = next(
        splitter_2.split(temp_df, temp_df["threat_label"])
    )

    return train_df, temp_df[val_idx], temp_df[test_idx]



def add_path_structure_features(df: pl.DataFrame) -> pl.DataFrame:
    """Extract structural features from request_path."""
    return df.with_columns([
        pl.col("request_path").str.lengths().alias("path_length"),
        pl.col("request_path").str.count_matches("/").alias("slash_count"),
        pl.col("request_path").str.count_matches(r"\.").alias("dot_count"),
        pl.col("request_path").str.count_matches(r"\.\.").alias("dotdot_count"),
        pl.col("request_path").str.count_matches(r"\?").alias("question_count"),
        pl.col("request_path").str.contains(r"\?").cast(pl.Int8).alias("has_query"),
        (pl.col("request_path").str.split("/").list.len() - 1).alias("path_depth"),
    ])

def add_ip_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add internal/external IP flags."""
    return df.with_columns([
        pl.col("source_ip").str.starts_with("192.168.").cast(pl.Int8).alias("src_internal_ip"),
        pl.col("destination_ip").str.starts_with("192.168.").cast(pl.Int8).alias("dst_internal_ip"),
    ])



def compute_ip_behavior(train_df: pl.DataFrame) -> pl.DataFrame:
    """Compute source IP behavior stats from TRAINING DATA ONLY."""
    return (
        train_df.groupby("source_ip")
        .agg([
            pl.count().alias("src_req_count"),
            (pl.col("threat_label") == "malicious").mean().alias("src_malicious_ratio"),
            (pl.col("threat_label") == "suspicious").mean().alias("src_suspicious_ratio"),
            pl.col("request_path").n_unique().alias("src_unique_paths"),
        ])
    )


def join_ip_behavior(df: pl.DataFrame, ip_stats: pl.DataFrame) -> pl.DataFrame:
    """Join IP behavior stats and safely handle unseen IPs."""
    return (
        df.join(ip_stats, on="source_ip", how="left")
        .with_columns([
            pl.col("src_req_count").fill_null(0),
            pl.col("src_malicious_ratio").fill_null(0.0),
            pl.col("src_suspicious_ratio").fill_null(0.0),
            pl.col("src_unique_paths").fill_null(0),
        ])
    )



def add_user_agent_features(df: pl.DataFrame) -> pl.DataFrame:
    """Abstract user agent strings into behavioral flags."""
    return df.with_columns([
        pl.col("user_agent").str.contains("Mozilla", literal=True).cast(pl.Int8).alias("ua_browser"),
        pl.col("user_agent").str.contains("curl", literal=True).cast(pl.Int8).alias("ua_cli"),
        pl.col("user_agent").str.contains("nmap|sqlmap", regex=True).cast(pl.Int8).alias("ua_scanner"),
    ])



def encode_and_scale(train_df, val_df, test_df):
    """Encode categorical features, scale numerics, encode target."""
    cat_cols = ["protocol", "log_type"]
    num_cols = [
        "bytes_transferred", "path_length", "slash_count",
        "dot_count", "dotdot_count", "question_count",
        "path_depth", "src_req_count",
        "src_malicious_ratio", "src_suspicious_ratio",
        "src_unique_paths",
    ]
    bin_cols = [
        "has_query", "src_internal_ip", "dst_internal_ip",
        "ua_browser", "ua_cli", "ua_scanner",
    ]

    # CATEGORICAL 
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(train_df.select(cat_cols).to_pandas())
    X_val_cat = encoder.transform(val_df.select(cat_cols).to_pandas())
    X_test_cat = encoder.transform(test_df.select(cat_cols).to_pandas())

    # NUMERIC 
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df.select(num_cols).to_pandas())
    X_val_num = scaler.transform(val_df.select(num_cols).to_pandas())
    X_test_num = scaler.transform(test_df.select(num_cols).to_pandas())

    # BINARY 
    X_train_bin = train_df.select(bin_cols).to_pandas().values
    X_val_bin = val_df.select(bin_cols).to_pandas().values
    X_test_bin = test_df.select(bin_cols).to_pandas().values

    # CONCAT 
    import numpy as np
    X_train = np.hstack([X_train_num, X_train_bin, X_train_cat])
    X_val = np.hstack([X_val_num, X_val_bin, X_val_cat])
    X_test = np.hstack([X_test_num, X_test_bin, X_test_cat])

    # TARGET 
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["threat_label"])
    y_val = label_encoder.transform(val_df["threat_label"])
    y_test = label_encoder.transform(test_df["threat_label"])

    artifacts = {
        "encoder": encoder,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_order": num_cols + bin_cols + list(encoder.get_feature_names_out(cat_cols)),
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, artifacts


def save_artifacts(artifacts: dict, output_dir: str = "artifacts"):
    """Persist preprocessing artifacts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        joblib.dump(obj, f"{output_dir}/{name}.joblib")


# MAIN PIPELINE (test run)

# if __name__ == "__main__":
#     file_path = "data/cybersecurity.csv"

#     lazy_df = load_data(file_path)
#     lazy_df = select_and_cast(lazy_df)
#     df = lazy_df.collect()

#     train_df, val_df, test_df = split_data(df)

#     for split in [train_df, val_df, test_df]:
#         split = add_path_structure_features(split)
#         split = add_ip_flags(split)
#         split = add_user_agent_features(split)

#     ip_stats = compute_ip_behavior(train_df)
#     train_df = join_ip_behavior(train_df, ip_stats)
#     val_df = join_ip_behavior(val_df, ip_stats)
#     test_df = join_ip_behavior(test_df, ip_stats)

#     X_train, X_val, X_test, y_train, y_val, y_test, artifacts = encode_and_scale(
#         train_df, val_df, test_df
#     )

#     save_artifacts(artifacts)
