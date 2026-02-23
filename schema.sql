-- =============================================================================
-- FRAUD DETECTION SYSTEM — SCHEMA DEFINITION
-- Architecture: staging → core → analytics (3-layer)
-- Database: PostgreSQL
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- SCHEMAS
-- ─────────────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ─────────────────────────────────────────────────────────────────────────────
-- LAYER 1: STAGING — Raw ingestion, minimal transformation
-- Purpose: Land raw CSV data exactly as-is; no business logic
-- ─────────────────────────────────────────────────────────────────────────────

DROP TABLE IF EXISTS staging.raw_transactions CASCADE;

CREATE TABLE staging.raw_transactions (
    step              INTEGER,          -- time unit (hour) in simulation
    type              TEXT,             -- PAYMENT, TRANSFER, CASH_OUT, etc.
    amount            NUMERIC,
    name_orig         TEXT,             -- sender ID
    old_balance_orig  NUMERIC,
    new_balance_orig  NUMERIC,
    name_dest         TEXT,             -- recipient ID
    old_balance_dest  NUMERIC,
    new_balance_dest  NUMERIC,
    is_fraud          INTEGER,          -- 0 or 1
    is_flagged_fraud  INTEGER,          -- system flag (for comparison)
    loaded_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Minimal index to speed up deduplication on load
CREATE INDEX IF NOT EXISTS idx_staging_nameOrig ON staging.raw_transactions(name_orig);


-- ─────────────────────────────────────────────────────────────────────────────
-- LAYER 2: CORE — Normalized relational model
-- Purpose: 3NF, FK constraints, referential integrity, business rules
-- ─────────────────────────────────────────────────────────────────────────────

-- Transaction type lookup (avoids repeating text in every row)
DROP TABLE IF EXISTS core.transaction_types CASCADE;
CREATE TABLE core.transaction_types (
    type_id   SERIAL PRIMARY KEY,
    type_name TEXT NOT NULL UNIQUE
);

INSERT INTO core.transaction_types (type_name)
VALUES ('PAYMENT'), ('TRANSFER'), ('CASH_OUT'), ('CASH_IN'), ('DEBIT')
ON CONFLICT DO NOTHING;


-- Users / Accounts (both senders and recipients)
DROP TABLE IF EXISTS core.accounts CASCADE;
CREATE TABLE core.accounts (
    account_id   SERIAL PRIMARY KEY,
    account_ref  TEXT NOT NULL UNIQUE,          -- e.g. "C1234567890"
    account_type CHAR(1) NOT NULL               -- 'C' = customer, 'M' = merchant
        CHECK (account_type IN ('C', 'M')),
    first_seen   TIMESTAMPTZ DEFAULT NOW(),
    last_seen    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_accounts_ref ON core.accounts(account_ref);


-- Transactions (normalized, FK-linked)
DROP TABLE IF EXISTS core.transactions CASCADE;
CREATE TABLE core.transactions (
    transaction_id      BIGSERIAL PRIMARY KEY,
    step                INTEGER NOT NULL CHECK (step >= 0),
    type_id             INTEGER NOT NULL REFERENCES core.transaction_types(type_id),
    amount              NUMERIC(18,2) NOT NULL CHECK (amount >= 0),

    sender_id           INTEGER NOT NULL REFERENCES core.accounts(account_id),
    sender_balance_pre  NUMERIC(18,2) CHECK (sender_balance_pre >= 0),
    sender_balance_post NUMERIC(18,2) CHECK (sender_balance_post >= 0),

    recipient_id        INTEGER NOT NULL REFERENCES core.accounts(account_id),
    recip_balance_pre   NUMERIC(18,2) CHECK (recip_balance_pre >= 0),
    recip_balance_post  NUMERIC(18,2) CHECK (recip_balance_post >= 0),

    is_fraud            BOOLEAN NOT NULL DEFAULT FALSE,
    is_flagged_fraud    BOOLEAN NOT NULL DEFAULT FALSE,

    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_txn_sender      ON core.transactions(sender_id);
CREATE INDEX IF NOT EXISTS idx_txn_recipient   ON core.transactions(recipient_id);
CREATE INDEX IF NOT EXISTS idx_txn_type        ON core.transactions(type_id);
CREATE INDEX IF NOT EXISTS idx_txn_step        ON core.transactions(step);
CREATE INDEX IF NOT EXISTS idx_txn_is_fraud    ON core.transactions(is_fraud);
-- Composite: fraud detection queries filter by type + fraud flag
CREATE INDEX IF NOT EXISTS idx_txn_type_fraud  ON core.transactions(type_id, is_fraud);
-- Composite: sender behavioral analysis
CREATE INDEX IF NOT EXISTS idx_txn_sender_step ON core.transactions(sender_id, step);


-- ─────────────────────────────────────────────────────────────────────────────
-- LAYER 3: ANALYTICS — Views, feature layer, reporting surfaces
-- Purpose: Pre-computed or view-based features for ML and dashboards
-- ─────────────────────────────────────────────────────────────────────────────

-- View: enriched transactions with human-readable fields
CREATE OR REPLACE VIEW analytics.v_transactions AS
SELECT
    t.transaction_id,
    t.step,
    tt.type_name,
    t.amount,
    s.account_ref  AS sender_ref,
    s.account_type AS sender_type,
    t.sender_balance_pre,
    t.sender_balance_post,
    r.account_ref  AS recipient_ref,
    r.account_type AS recipient_type,
    t.recip_balance_pre,
    t.recip_balance_post,
    t.is_fraud,
    t.is_flagged_fraud
FROM core.transactions t
JOIN core.transaction_types tt ON t.type_id = tt.type_id
JOIN core.accounts s           ON t.sender_id = s.account_id
JOIN core.accounts r           ON t.recipient_id = r.account_id;


-- Materialized view: sender behavioral features (refreshed on pipeline run)
DROP MATERIALIZED VIEW IF EXISTS analytics.mv_sender_features;
CREATE MATERIALIZED VIEW analytics.mv_sender_features AS
WITH sender_stats AS (
    SELECT
        sender_id,
        COUNT(*)                                                   AS total_txn_count,
        SUM(amount)                                                AS total_amount,
        AVG(amount)                                                AS avg_amount,
        STDDEV(amount)                                             AS stddev_amount,
        MAX(amount)                                                AS max_amount,
        MIN(amount)                                                AS min_amount,
        COUNT(*) FILTER (WHERE is_fraud)                          AS fraud_count,
        ROUND(
            COUNT(*) FILTER (WHERE is_fraud)::NUMERIC / NULLIF(COUNT(*), 0), 4
        )                                                          AS fraud_rate,
        COUNT(DISTINCT recipient_id)                              AS unique_recipients,
        MAX(step) - MIN(step)                                     AS active_duration_steps
    FROM core.transactions
    GROUP BY sender_id
)
SELECT
    ss.*,
    a.account_ref
FROM sender_stats ss
JOIN core.accounts a ON ss.sender_id = a.account_id;

CREATE UNIQUE INDEX ON analytics.mv_sender_features(sender_id);


-- Materialized view: ML feature set (one row per transaction)
DROP MATERIALIZED VIEW IF EXISTS analytics.mv_ml_features;
CREATE MATERIALIZED VIEW analytics.mv_ml_features AS
WITH base AS (
    SELECT
        t.transaction_id,
        t.step,
        tt.type_name,
        t.amount,
        t.sender_id,
        t.sender_balance_pre,
        t.sender_balance_post,
        t.recip_balance_pre,
        t.recip_balance_post,
        t.is_fraud,

        -- ── Feature: balance deltas (anomaly signal)
        -- Fraudulent TRANSFER/CASH_OUT often drains sender to 0
        (t.sender_balance_pre - t.sender_balance_post)           AS sender_balance_drop,
        (t.recip_balance_post - t.recip_balance_pre)             AS recipient_balance_gain,

        -- ── Feature: balance mismatch (data integrity + fraud signal)
        -- Amount transferred doesn't match recipient balance change → suspicious
        ABS(t.amount - (t.recip_balance_post - t.recip_balance_pre)) AS dest_balance_mismatch,

        -- ── Feature: sender drained to zero
        CASE WHEN t.sender_balance_post = 0 THEN 1 ELSE 0 END   AS sender_drained,

        -- ── Feature: recipient pre-balance was zero (new/empty account)
        CASE WHEN t.recip_balance_pre = 0 THEN 1 ELSE 0 END      AS dest_was_empty,

        -- ── Window feature: sender's transaction rank by amount (high = anomalous)
        RANK() OVER (
            PARTITION BY t.sender_id
            ORDER BY t.amount DESC
        )                                                          AS sender_amount_rank,

        -- ── Window feature: running total per sender
        SUM(t.amount) OVER (
            PARTITION BY t.sender_id
            ORDER BY t.step
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                          AS sender_cumulative_amount,

        -- ── Window feature: count of sender txns in same step (velocity)
        COUNT(*) OVER (
            PARTITION BY t.sender_id, t.step
        )                                                          AS sender_txns_same_step,

        -- ── Window feature: amount z-score vs sender history
        CASE
            WHEN STDDEV(t.amount) OVER (PARTITION BY t.sender_id) = 0 THEN 0
            ELSE (t.amount - AVG(t.amount) OVER (PARTITION BY t.sender_id))
                 / STDDEV(t.amount) OVER (PARTITION BY t.sender_id)
        END                                                        AS sender_amount_zscore,

        -- ── Window feature: previous transaction amount (lag)
        LAG(t.amount) OVER (
            PARTITION BY t.sender_id ORDER BY t.step
        )                                                          AS sender_prev_amount,

        -- ── Sender aggregate features (from mv_sender_features join)
        sf.total_txn_count     AS sender_total_txns,
        sf.avg_amount          AS sender_avg_amount,
        sf.fraud_rate          AS sender_historical_fraud_rate,
        sf.unique_recipients   AS sender_unique_recipients

    FROM core.transactions t
    JOIN core.transaction_types tt ON t.type_id = tt.type_id
    JOIN analytics.mv_sender_features sf ON t.sender_id = sf.sender_id
)
SELECT * FROM base;

CREATE INDEX ON analytics.mv_ml_features(transaction_id);
CREATE INDEX ON analytics.mv_ml_features(is_fraud);


-- View: fraud rate by transaction type (dashboard)
CREATE OR REPLACE VIEW analytics.v_fraud_by_type AS
SELECT
    tt.type_name,
    COUNT(*)                                           AS total_transactions,
    COUNT(*) FILTER (WHERE t.is_fraud)                 AS fraud_count,
    ROUND(
        COUNT(*) FILTER (WHERE t.is_fraud)::NUMERIC
        / NULLIF(COUNT(*), 0) * 100, 2
    )                                                  AS fraud_rate_pct,
    SUM(t.amount)                                      AS total_volume,
    SUM(t.amount) FILTER (WHERE t.is_fraud)            AS fraud_volume
FROM core.transactions t
JOIN core.transaction_types tt ON t.type_id = tt.type_id
GROUP BY tt.type_name
ORDER BY fraud_rate_pct DESC;


-- View: fraud trend over time (hourly step)
CREATE OR REPLACE VIEW analytics.v_fraud_trend AS
SELECT
    step,
    COUNT(*)                                      AS total_txns,
    COUNT(*) FILTER (WHERE is_fraud)              AS fraud_txns,
    SUM(amount)                                   AS total_volume,
    SUM(amount) FILTER (WHERE is_fraud)           AS fraud_volume,
    ROUND(
        COUNT(*) FILTER (WHERE is_fraud)::NUMERIC
        / NULLIF(COUNT(*), 0) * 100, 3
    )                                             AS fraud_rate_pct
FROM core.transactions
GROUP BY step
ORDER BY step;


-- View: risky senders (top fraud contributors)
CREATE OR REPLACE VIEW analytics.v_risky_senders AS
SELECT
    a.account_ref,
    sf.total_txn_count,
    sf.fraud_count,
    sf.fraud_rate,
    sf.total_amount,
    sf.avg_amount,
    sf.unique_recipients
FROM analytics.mv_sender_features sf
JOIN core.accounts a ON sf.sender_id = a.account_id
WHERE sf.fraud_count > 0
ORDER BY sf.fraud_rate DESC, sf.fraud_count DESC;
