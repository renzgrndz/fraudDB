-- =============================================================================
-- FRAUD DETECTION SYSTEM — ANALYTICAL & VALIDATION QUERIES
-- =============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 1: REFERENTIAL INTEGRITY VALIDATION
-- ─────────────────────────────────────────────────────────────────────────────

-- 1.1 Every transaction references a valid transaction type
SELECT COUNT(*) AS orphaned_type_refs
FROM core.transactions t
LEFT JOIN core.transaction_types tt ON t.type_id = tt.type_id
WHERE tt.type_id IS NULL;
-- EXPECTED: 0

-- 1.2 Every sender_id exists in core.accounts
SELECT COUNT(*) AS orphaned_senders
FROM core.transactions t
LEFT JOIN core.accounts a ON t.sender_id = a.account_id
WHERE a.account_id IS NULL;
-- EXPECTED: 0

-- 1.3 Every recipient_id exists in core.accounts
SELECT COUNT(*) AS orphaned_recipients
FROM core.transactions t
LEFT JOIN core.accounts a ON t.recipient_id = a.account_id
WHERE a.account_id IS NULL;
-- EXPECTED: 0

-- 1.4 No negative amounts (CHECK constraint backup validation)
SELECT COUNT(*) AS negative_amounts
FROM core.transactions
WHERE amount < 0;
-- EXPECTED: 0

-- 1.5 Staging row count matches core row count (pipeline completeness)
SELECT
    (SELECT COUNT(*) FROM staging.raw_transactions) AS staging_rows,
    (SELECT COUNT(*) FROM core.transactions)        AS core_rows,
    (SELECT COUNT(*) FROM staging.raw_transactions)
    - (SELECT COUNT(*) FROM core.transactions)      AS discrepancy;
-- EXPECTED discrepancy: 0

-- 1.6 Fraud label distribution (sanity check — fraud should be ~0.1%)
SELECT
    is_fraud,
    COUNT(*)                                         AS count,
    ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER () * 100, 3) AS pct
FROM core.transactions
GROUP BY is_fraud;

-- 1.7 Validate normalization: no duplicate (step, amount, sender, recipient) combos
SELECT step, amount, sender_id, recipient_id, COUNT(*) AS dupes
FROM core.transactions
GROUP BY step, amount, sender_id, recipient_id
HAVING COUNT(*) > 1
LIMIT 10;


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 2: ANALYTICAL QUERIES (Window Functions)
-- ─────────────────────────────────────────────────────────────────────────────

-- 2.1 Fraud rate by transaction type with ranking
SELECT
    tt.type_name,
    COUNT(*)                                              AS total_txns,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)           AS fraud_txns,
    ROUND(
        AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END) * 100, 3
    )                                                     AS fraud_rate_pct,
    RANK() OVER (
        ORDER BY AVG(CASE WHEN t.is_fraud THEN 1.0 ELSE 0.0 END) DESC
    )                                                     AS fraud_rank
FROM core.transactions t
JOIN core.transaction_types tt ON t.type_id = tt.type_id
GROUP BY tt.type_name
ORDER BY fraud_rate_pct DESC;


-- 2.2 Rolling 10-step fraud rate (moving average)
WITH hourly AS (
    SELECT
        step,
        COUNT(*) FILTER (WHERE is_fraud)::NUMERIC / NULLIF(COUNT(*), 0) AS step_fraud_rate
    FROM core.transactions
    GROUP BY step
)
SELECT
    step,
    step_fraud_rate,
    AVG(step_fraud_rate) OVER (
        ORDER BY step
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ) AS rolling_10_fraud_rate
FROM hourly
ORDER BY step;


-- 2.3 Top 20 riskiest senders by fraud count
SELECT
    a.account_ref,
    COUNT(*)                              AS total_txns,
    COUNT(*) FILTER (WHERE t.is_fraud)    AS fraud_txns,
    SUM(t.amount) FILTER (WHERE t.is_fraud) AS fraud_volume,
    ROUND(
        COUNT(*) FILTER (WHERE t.is_fraud)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2
    )                                     AS fraud_rate_pct,
    DENSE_RANK() OVER (
        ORDER BY COUNT(*) FILTER (WHERE t.is_fraud) DESC
    )                                     AS rank
FROM core.transactions t
JOIN core.accounts a ON t.sender_id = a.account_id
GROUP BY a.account_ref
HAVING COUNT(*) FILTER (WHERE t.is_fraud) > 0
ORDER BY fraud_txns DESC
LIMIT 20;


-- 2.4 Sender anomaly detection: transactions 3+ std deviations above personal mean
WITH sender_stats AS (
    SELECT
        sender_id,
        AVG(amount) AS mean_amt,
        STDDEV(amount) AS std_amt
    FROM core.transactions
    GROUP BY sender_id
)
SELECT
    t.transaction_id,
    a.account_ref,
    t.amount,
    ss.mean_amt,
    ss.std_amt,
    ROUND((t.amount - ss.mean_amt) / NULLIF(ss.std_amt, 0), 2) AS z_score,
    t.is_fraud
FROM core.transactions t
JOIN sender_stats ss ON t.sender_id = ss.sender_id
JOIN core.accounts a ON t.sender_id = a.account_id
WHERE ss.std_amt > 0
  AND (t.amount - ss.mean_amt) / ss.std_amt > 3
ORDER BY z_score DESC
LIMIT 50;


-- 2.5 Destination account analysis: accounts that ONLY receive (laundering pattern)
WITH recv AS (
    SELECT recipient_id, COUNT(*) AS recv_count, SUM(amount) AS recv_total
    FROM core.transactions GROUP BY recipient_id
),
sent AS (
    SELECT sender_id, COUNT(*) AS sent_count
    FROM core.transactions GROUP BY sender_id
)
SELECT
    a.account_ref,
    r.recv_count,
    r.recv_total,
    COALESCE(s.sent_count, 0) AS sent_count
FROM recv r
JOIN core.accounts a ON r.recipient_id = a.account_id
LEFT JOIN sent s ON r.recipient_id = s.sender_id
WHERE COALESCE(s.sent_count, 0) = 0
  AND r.recv_count > 5
ORDER BY r.recv_total DESC
LIMIT 20;


-- 2.6 Percentile distribution of transaction amounts by type
SELECT
    tt.type_name,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) AS p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount) AS p50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) AS p75,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount) AS p90,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount) AS p99,
    MAX(amount) AS max_amount
FROM core.transactions t
JOIN core.transaction_types tt ON t.type_id = tt.type_id
GROUP BY tt.type_name;


-- 2.7 Burst detection: senders with > 5 transactions in a single step
SELECT
    a.account_ref,
    t.step,
    COUNT(*) AS txn_count,
    SUM(t.amount) AS step_volume
FROM core.transactions t
JOIN core.accounts a ON t.sender_id = a.account_id
GROUP BY a.account_ref, t.step
HAVING COUNT(*) > 5
ORDER BY txn_count DESC;


-- 2.8 Balance inconsistency: sender balance_post doesn't match next txn balance_pre
WITH ordered AS (
    SELECT
        transaction_id,
        sender_id,
        step,
        sender_balance_pre,
        sender_balance_post,
        LAG(sender_balance_post) OVER (
            PARTITION BY sender_id ORDER BY step, transaction_id
        ) AS prev_balance_post
    FROM core.transactions
)
SELECT
    transaction_id,
    sender_id,
    step,
    sender_balance_pre,
    prev_balance_post,
    ABS(sender_balance_pre - prev_balance_post) AS inconsistency
FROM ordered
WHERE prev_balance_post IS NOT NULL
  AND ABS(sender_balance_pre - prev_balance_post) > 0.01
ORDER BY inconsistency DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────────────────────
-- SECTION 3: ML FEATURE VALIDATION
-- ─────────────────────────────────────────────────────────────────────────────

-- 3.1 Feature completeness check
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE sender_amount_zscore IS NULL)  AS null_zscores,
    COUNT(*) FILTER (WHERE sender_prev_amount IS NULL)    AS null_lag_amount,
    COUNT(*) FILTER (WHERE sender_total_txns IS NULL)     AS null_sender_stats
FROM analytics.mv_ml_features;

-- 3.2 Class balance in ML feature set
SELECT
    is_fraud,
    COUNT(*)                                                      AS count,
    ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER () * 100, 3)    AS pct
FROM analytics.mv_ml_features
GROUP BY is_fraud;

-- 3.3 Feature correlation sanity: fraud vs sender_drained
SELECT
    sender_drained,
    COUNT(*)                                     AS count,
    COUNT(*) FILTER (WHERE is_fraud)             AS fraud_count,
    ROUND(
        COUNT(*) FILTER (WHERE is_fraud)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2
    )                                            AS fraud_rate_pct
FROM analytics.mv_ml_features
GROUP BY sender_drained
ORDER BY sender_drained;

-- 3.4 Feature correlation sanity: fraud vs dest_was_empty
SELECT
    dest_was_empty,
    COUNT(*)                                     AS count,
    COUNT(*) FILTER (WHERE is_fraud)             AS fraud_count,
    ROUND(
        COUNT(*) FILTER (WHERE is_fraud)::NUMERIC / NULLIF(COUNT(*), 0) * 100, 2
    )                                            AS fraud_rate_pct
FROM analytics.mv_ml_features
GROUP BY dest_was_empty
ORDER BY dest_was_empty;
