"""Dataset and model constants."""

from __future__ import annotations

CICIDS_FILE_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

CICIDS_BENIGN_LABEL = "BENIGN"
CICIDS_CLASSES = [
    CICIDS_BENIGN_LABEL,
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack Brute Force",
    "Web Attack Sql Injection",
    "Web Attack XSS",
]
CICIDS_DROP_COLUMNS = {"Flow ID", "Source IP", "Destination IP", "Timestamp"}
CICIDS_FORCED_CATEGORICAL = {"Protocol", "Src Port Bucket", "Dst Port Bucket"}
CICIDS_CANONICAL_LABELS = {
    "Web Attack \ufeff Brute Force": "Web Attack Brute Force",
    "Web Attack \ufeff Sql Injection": "Web Attack Sql Injection",
    "Web Attack \ufeff XSS": "Web Attack XSS",
    "Web Attack � Brute Force": "Web Attack Brute Force",
    "Web Attack � Sql Injection": "Web Attack Sql Injection",
    "Web Attack � XSS": "Web Attack XSS",
}

UNSW_BENIGN_LABEL = "Normal"
UNSW_CLASSES = [
    UNSW_BENIGN_LABEL,
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]
UNSW_COMMON_ATTACKS = {"Generic", "Exploits", "Fuzzers"}
UNSW_RARE_ATTACKS = {
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
    "Reconnaissance",
    "DoS",
}
UNSW_WARMUP_POOL = {UNSW_BENIGN_LABEL, *UNSW_COMMON_ATTACKS}
UNSW_DROP_COLUMNS = {"id", "\ufeffid", "attack_cat", "label"}
UNSW_FORCED_CATEGORICAL = {"proto", "service", "state"}
UNSW_STAGE_ORDER = [
    "warmup",
    "benign_drift",
    "rare_attack",
    "recurrent_common",
]

DEFAULT_OUTPUT_DIR = "outputs"
