import sqlite3
from datetime import datetime

DB_PATH = "patients.db"


# -------------------------
# CREATE CONNECTION (SAFE)
# -------------------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


# -------------------------
# INIT DATABASE
# -------------------------
def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT,
        name TEXT,
        age INTEGER,
        image_path TEXT,
        area REAL,
        perimeter REAL,
        roughness REAL,
        volume REAL,
        max_depth REAL,
        mean_depth REAL,
        classification TEXT,
        confidence REAL,
        risk TEXT,
        report TEXT,
        date TEXT,
        wolfram_analysis TEXT
    )
    """)
    
    # Run migrations for existing databases that don't have the wolfram_analysis column
    try:
        cursor.execute("ALTER TABLE patients ADD COLUMN wolfram_analysis TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    conn.commit()
    conn.close()


# -------------------------
# SAVE PATIENT DATA
# -------------------------
def save_patient(data):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO patients (
        patient_id, name, age, image_path,
        area, perimeter, roughness, volume,
        max_depth, mean_depth,
        classification, confidence, risk,
        report, date, wolfram_analysis
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("patient_id"),
        data.get("name"),
        data.get("age"),
        data.get("image_path"),
        data.get("area"),
        data.get("perimeter"),
        data.get("roughness"),
        data.get("volume"),
        data.get("max_depth"),
        data.get("mean_depth"),
        data.get("classification"),
        data.get("confidence"),
        data.get("risk"),
        data.get("report"),
        data.get("date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        data.get("wolfram_analysis")
    ))

    conn.commit()
    conn.close()


# -------------------------
# UPDATE REPORT (VERY IMPORTANT)
# -------------------------
def update_report(patient_id, report):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE patients
    SET report = ?
    WHERE rowid = (
        SELECT rowid FROM patients
        WHERE patient_id = ?
        ORDER BY date DESC
        LIMIT 1
    )
    """, (report, patient_id))

    conn.commit()
    conn.close()


# -------------------------
# FETCH PATIENT HISTORY
# -------------------------
def get_patient_history(patient_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM patients
    WHERE patient_id = ?
    ORDER BY date DESC
    """, (patient_id,))

    rows = cursor.fetchall()
    conn.close()

    return rows