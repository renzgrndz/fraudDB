-- =============================================================================
-- FRAUD DETECTION SYSTEM — DATA LOADING PIPELINE
-- Loads PaySim CSV into staging, then promotes to core with normalization
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 1: Load raw CSV into staging
-- Run from psql:  \COPY staging.raw_transactions(...) FROM 'paysim_100k.csv' CSV HEADER
-- Or use Python SQLAlchemy COPY for larger files
-- ─────────────────────────────────────────────────────────────────────────────

-- Example psql command (adjust path):
-- \COPY staging.raw_transactions(step, type, amount, name_orig, old_balance_orig,
--   new_balance_orig, name_dest, old_balance_dest, new_balance_dest, is_fraud, is_flagged_fraud)
-- FROM '/data/paysim_100k.csv' WITH (FORMAT CSV, HEADER TRUE);

-- Verify load
SELECT COUNT(*) AS staged_rows FROM staging.raw_transactions;
SELECT type, COUNT(*) FROM staging.raw_transactions GROUP BY type ORDER BY COUNT(*) DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 2: Populate core.transaction_types from staging (if new types appear)
-- ─────────────────────────────────────────────────────────────────────────────
INSERT INTO core.transaction_types (type_name)
SELECT DISTINCT UPPER(TRIM(type))
FROM staging.raw_transactions
ON CONFLICT (type_name) DO NOTHING;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 3: Populate core.accounts (union of senders + recipients)
-- account_type derived from ID prefix: 'C' = Customer, 'M' = Merchant
-- ─────────────────────────────────────────────────────────────────────────────
INSERT INTO core.accounts (account_ref, account_type)
SELECT DISTINCT
    account_ref,
    CASE WHEN account_ref LIKE 'M%' THEN 'M' ELSE 'C' END AS account_type
FROM (
    SELECT name_orig AS account_ref FROM staging.raw_transactions
    UNION
    SELECT name_dest AS account_ref FROM staging.raw_transactions
) all_accounts
ON CONFLICT (account_ref) DO NOTHING;

-- Update last_seen (idempotent refresh)
UPDATE core.accounts a
SET last_seen = NOW()
WHERE EXISTS (
    SELECT 1 FROM staging.raw_transactions r
    WHERE r.name_orig = a.account_ref OR r.name_dest = a.account_ref
);

SELECT COUNT(*) AS total_accounts FROM core.accounts;
SELECT account_type, COUNT(*) FROM core.accounts GROUP BY account_type;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 4: Populate core.transactions (normalized, FK-resolved)
-- ─────────────────────────────────────────────────────────────────────────────
INSERT INTO core.transactions (
    step, type_id, amount,
    sender_id, sender_balance_pre, sender_balance_post,
    recipient_id, recip_balance_pre, recip_balance_post,
    is_fraud, is_flagged_fraud
)
SELECT
    r.step,
    tt.type_id,
    r.amount,
    s.account_id                        AS sender_id,
    r.old_balance_orig                  AS sender_balance_pre,
    r.new_balance_orig                  AS sender_balance_post,
    rec.account_id                      AS recipient_id,
    r.old_balance_dest                  AS recip_balance_pre,
    r.new_balance_dest                  AS recip_balance_post,
    r.is_fraud = 1                      AS is_fraud,
    r.is_flagged_fraud = 1              AS is_flagged_fraud
FROM staging.raw_transactions r
JOIN core.transaction_types tt ON tt.type_name = UPPER(TRIM(r.type))
JOIN core.accounts s           ON s.account_ref = r.name_orig
JOIN core.accounts rec         ON rec.account_ref = r.name_dest;

SELECT COUNT(*) AS loaded_transactions FROM core.transactions;


-- ─────────────────────────────────────────────────────────────────────────────
-- STEP 5: Refresh materialized views
-- ─────────────────────────────────────────────────────────────────────────────
REFRESH MATERIALIZED VIEW analytics.mv_sender_features;
REFRESH MATERIALIZED VIEW analytics.mv_ml_features;

SELECT 'Pipeline complete. Materialized views refreshed.' AS status;
