# Synthetic Data Dictionary

Six tables extend the original customer dataset. Account IDs remain unchanged during renewal; savings_cycles preserves each term separately.

**Snapshot boundary:** 1 October 2026, 00:00 (exclusive). Snapshot balances represent the state immediately before this boundary.
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
| renewal_type | STRING, nullable | Maturity instruction: auto starts a new cycle using principal plus posted interest; manual pays principal and interest to the payout account and closes the savings account | manual, auto; NULL for everyday |
| payout_account_id | STRING, nullable | Everyday account receiving savings principal and interest at settlement. Must remain open until settlement. | Existing everyday account owned by the same customer for savings; NULL for everyday |

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
| transfer_id | STRING, nullable | Links both sides of a transfer between modeled accounts | Same non-empty ID for exactly two successful transfer records; NULL for unpaired transactions. |

### Transaction types

| Type | Meaning |
|---|---|
| external_transfer | Transfer to/from another customer's account or an account outside the modeled bank. Direction indicates money received (in) or sent (out). |
| internal_transfer | Transfer between two modeled accounts owned by the same customer. |
| payment | Payment for goods or services, including recurring bills and subscriptions. Direction: out. |
| cash_deposit | Cash paid into an account. Direction: in. |
| cash_withdrawal | Cash withdrawn from an account. Direction: out. |
| interest_credit | Interest posted by the bank. Direction: in; channel: system. |
| bank_fee | Fee charged by the bank. Direction: out; channel: system. |

**Transfer rules:**
- Transfers between modeled accounts have two successful records with the same transfer_id, amount and timestamp, opposite directions, and different account_id values.
- Same customer: internal_transfer. Different customers: external_transfer.
- For external_transfer without transfer_id, the counterparty is assumed to be outside the modeled bank. Only the modeled account's side is recorded.
- System-channel incoming external transfers are passive receipts, including simulated income and initial funding; they do not count as customer-initiated activity.

## savings_cycles

> One row per savings term. The same account_id can have several cycles. Key: (account_id, started_at).

| Field | Type | Description / generation rule | Valid values / range |
|---|---|---|---|
| account_id | STRING | Savings account to which this cycle belongs (FK). | Existing accounts.account_id with account_type = savings |
| started_at | TIMESTAMP | First deposit time for the first cycle, or renewal time for later cycles. Preserve cycles overlapping W. | created_at ≤ started_at < snapshot boundary; (account_id, started_at) must be unique |
| term_months | INT | Agreed term for this cycle. | 1,3,6 |
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

## customer_context

One row per customer. Provides the dates needed to calculate observation periods and exposure-adjusted features.

| Field | Type | Description | Valid values / range |
|---|---|---|---|
| customer_id | BIGINT | Customer identifier (FK). | Existing customer_id; unique |
| relationship_start_date | TIMESTAMP | Simulated relationship start, consistent with tenure. | Before snapshot |
| observation_start | TIMESTAMP | Later of relationship_start_date and 1 April 2026. | 1 April 2026 ≤ value < snapshot |
| snapshot_boundary | TIMESTAMP | Exclusive end of the observation window. | 1 October 2026, 00:00 |

**Usage:** Calculate exposure from the later of the feature window start and relationship_start_date. History before the customer joined is unobserved, not zero activity.

**Generation parameters:** Distributions, probabilities, rates and the random seed are documented in the generator notebook and generation_manifest.json.

**Balance rule:** Total account balances per customer at snapshot equal the original customer balance. Closed accounts have zero balance.

**Limit:** All behavioral histories and relationship dates are simulated. The dataset does not model loans, credit-card debt, or complete CLV.
