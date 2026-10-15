# Synthetic Data Dictionary

Five tables extend the original customer dataset. Account IDs remain unchanged during renewal; savings_cycles preserves each term separately.

**Snapshot:** end of 30/9/2026
**Event window (W):** 1/4–30/9/2026, inclusive
**Currency:** CU for every account and transaction; stored once in configuration, not per row. These are project assumptions.

## accounts

> One row per everyday or savings account. Auto renewals keep the same account_id. These accounts do not reconstruct every product counted in products_number.

| Field | Type | Description / generation rule | Valid values / range |
|---|---|---|---|
| account_id | STRING | Unique generated account ID (PK). | Non-empty, unique; e.g. ACC000001 |
| customer_id | BIGINT | Account owner; copied from the original data (FK). | Existing original customer_id |
| account_type | STRING | Simulated deposit-account type. | everyday, savings |
| created_at | TIMESTAMP | Generated opening time, not earlier than the customer's simulated relationship start. | Relationship start ≤ created_at < snapshot boundary |
| closed_at | TIMESTAMP, nullable | Actual closure time; renewal does not close the account. Closing savings alone is not customer churn. | created_at ≤ closed_at < snapshot boundary; NULL if still open |
| balance | DECIMAL(18,2) | Posted balance at snapshot. Open savings hold the current cycle's principal only; unposted interest is excluded. Zero when closed. | ≥ 0 CU; two decimal places |
| renewal_type | STRING, nullable | Maturity instruction, fixed per account for simulation. Manual renewal returns savings and interests to payout account | manual, auto; NULL for everyday |
| payout_account_id | STRING | Everyday accounts that receive savings + interests after payout. Need guardrail so that payout account cannot be closed when there exists a saving account linked to it | everyday account of the same customer, NULL for everyday

## login

> One row per login attempt.

| Field | Type | Description / generation rule | Valid values / range |
|---|---|---|---|
| login_id | STRING | Unique generated login ID (PK). | Non-empty, unique; e.g. LOG000001 |
| customer_id | BIGINT | Customer attempting login (FK). | Existing original customer_id |
| sign_in_datetime | TIMESTAMP | Sampled from customer login frequency and time patterns. | Within W, on/after relationship start |
| login_status | STRING | Sampled login result. Only successful logins enter engagement counts. Failed login due to an incorrect password, failed authentication, a locked account, or a system error.  | success, failed |
| device_type | STRING | Sampled from customer device preferences. | android, ios, desktop |

## transactions

> One row per account-side transaction record. A transfer between two modeled accounts has two linked records.

| Field | Type | Description / generation rule | Valid values / range |
|---|---|---|---|
| transaction_id | STRING | Unique generated record ID (PK). | Non-empty, unique; e.g. TXN000001 |
| account_id | STRING | Account affected by this record (FK); join accounts to obtain customer_id. | Existing accounts.account_id |
| transaction_datetime | TIMESTAMP | Sampled from transaction frequency and time patterns. | Within W, on/after opening and on/before closure |
| amount | DECIMAL(18,2) | Positive transaction value; sampled by type and available funds. | > 0 CU; two decimal places |
| direction | STRING | Money movement relative to account_id. | in, out |
| transaction_type | STRING | Simulated transaction category. | external_transfer, internal_transfer, payment, cash_deposit, cash_withdrawal, interest_credit, bank_fee |
| transaction_status | STRING | Sampled outcome; only success changes balances. | success, failed, cancelled |
| channel | STRING | Sampled channel compatible with transaction type; system marks automatic postings. | mobile_app, atm, branch, system |
| transfer_id | STRING, nullable | Links two sides of the same internal transfer. | Same non-empty ID for exactly two successful internal-transfer records; otherwise NULL |

## savings_cycles

> One row per savings term. The same account_id can have several cycles. Key: (account_id, started_at).

| Field | Type | Description / generation rule | Valid values / range |
|---|---|---|---|
| account_id | STRING | Savings account to which this cycle belongs (FK). | Existing accounts.account_id with account_type = savings |
| started_at | TIMESTAMP | First deposit time or actual renewal time. Preserve all cycles overlapping W; the first may start before W. | Account opening ≤ started_at < snapshot boundary; unique per account |
| term_months | INT | Agreed term for this cycle. | Positive integer |
| init_balance | DECIMAL(18,2) | Money committed at cycle start. On renewal, previous balance plus posted interest. | > 0 CU; two decimal places |
| interest_rate | DECIMAL | Annual maturity rate agreed for this cycle; generated from documented parameters. | 0–1 (0.04 = 4% per year) |
| early_withdrawal_rate | DECIMAL | Annual early-closure rate, assumed fixed within the cycle. | 0 ≤ rate ≤ interest_rate |

## complaints

> One row per customer complaint.

| Field | Type | Description / generation rule | Valid values / range |
|---|---|---|---|
| complaint_id | STRING | Unique generated complaint ID (PK). | Non-empty, unique; e.g. CMP000001 |
| customer_id | BIGINT | Customer making the complaint (FK). | Existing original customer_id |
| complaint_datetime | TIMESTAMP | Sampled from a documented complaint rate. | Within W, on/after relationship start |
| resolved_datetime | TIMESTAMP, nullable | Generated from a resolution delay. NULL if unresolved at snapshot. | complaint_datetime ≤ resolved_datetime < snapshot boundary; otherwise NULL |

**Limit:** This dictionary specifies fields and constraints. Generation parameters still need to be chosen and documented. It does not model loans, credit-card debt, or complete CLV.

