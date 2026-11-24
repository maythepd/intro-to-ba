"""Construct new customer profiles and event histories from matched templates."""
import numpy as np
import pandas as pd

from segmentation.data import DATES

PROFILE_FIELDS = ("age", "credit_score", "credit_card", "estimated_salary")
MONEY_CHANGE = (.04, .12)
TIME_WARP_SECONDS = (1200, 2400)


def match_profiles(customers, rng):
    """Rotate demographic templates within strata that support similar behavior."""
    population = customers.sort_values("customer_id").set_index("customer_id")
    strata = population.assign(
        tenure_band=population.tenure // 3,
        salary_band=np.searchsorted(population.estimated_salary.quantile([.2, .4, .6, .8]),
                                    population.estimated_salary),
        has_balance=population.balance.gt(0),
    )
    keys = ["country", "gender", "active_member", "products_number", "tenure_band", "salary_band", "has_balance"]
    mapping = {}
    for group in strata.groupby(keys, sort=True).indices.values():
        ids = rng.permutation(population.index.to_numpy()[group])
        mapping.update(zip(ids, np.roll(ids, 1)))
    return pd.Series(mapping, dtype="int64")


def warp_clock(times, seconds):
    """Keep calendar days and ordering, including times shared by linked events."""
    offset = (times - times.dt.normalize()).dt.total_seconds() / 86400
    return times + pd.to_timedelta(seconds * np.sin(np.pi * offset), unit="s")


def scoped_id(customers, template_keys):
    return "D" + customers.astype(str) + "-" + template_keys.astype("string")


def synthesize(tables, demo_ids, history_ids, profile_ids, warp_seconds, rng):
    """Create independent demo identities and apply coherent transformations to their templates."""
    size = len(demo_ids)
    cohort = pd.DataFrame({
        "customer_id": demo_ids, "template_customer": history_ids,
        "money_scale": 1 + rng.choice((-1, 1), size) * rng.uniform(*MONEY_CHANGE, size=size),
    })
    cohort["occurrence"] = cohort.groupby("template_customer", sort=False).cumcount()
    customers = tables["kaggle_bank_churn"].set_index("customer_id")
    base = customers.loc[history_ids].set_axis(demo_ids).rename_axis("customer_id")
    base[list(PROFILE_FIELDS)] = customers.loc[profile_ids, list(PROFILE_FIELDS)].set_axis(base.index)
    base["estimated_salary"] = (base.estimated_salary.to_numpy() * cohort.money_scale).round(2).to_numpy()
    # Keep zero salary valid while ensuring that the static profile is still novel.
    zero_salary = base.estimated_salary.eq(0)
    score = base.loc[zero_salary, "credit_score"]
    base.loc[zero_salary, "credit_score"] = np.where(score.le(845), score + 5, score - 5)

    def customer_rows(name):
        template = tables[name]
        selected = template.loc[template.customer_id.isin(history_ids)]
        return selected.rename(columns={"customer_id": "template_customer"}).merge(
            cohort, on="template_customer", validate="many_to_many")

    accounts = customer_rows("accounts").rename(columns={"account_id": "template_account"})
    accounts["account_id"] = scoped_id(accounts.customer_id, accounts.template_account)
    accounts["payout_account_id"] = scoped_id(accounts.customer_id, accounts.payout_account_id)
    accounts["balance"] = (accounts.balance * accounts.money_scale).round(2)
    open_balance = accounts.loc[accounts.closed_at.isna()].groupby("customer_id").balance.sum()
    base["balance"] = open_balance.reindex(base.index, fill_value=0).round(2)

    def account_rows(name):
        template = tables[name]
        selected = template.loc[template.account_id.isin(accounts.template_account)]
        links = accounts[["template_account", "account_id", "customer_id", "money_scale", "occurrence"]]
        return selected.rename(columns={"account_id": "template_account"}).merge(
            links, on="template_account", validate="many_to_many")

    tx = account_rows("transactions")
    tx["transaction_id"] = scoped_id(tx.customer_id, tx.transaction_id)
    internal = tx.transaction_type.eq("internal_transfer") & tx.transfer_id.notna()
    external = tx.transaction_type.eq("external_transfer") & tx.transfer_id.notna()
    tx["transfer_id"] = tx.transfer_id.astype("string")
    tx.loc[internal, "transfer_id"] = scoped_id(tx.loc[internal, "customer_id"], tx.loc[internal, "transfer_id"])
    # Preserve both sides if the two counterparties are in this batch. A missing
    # counterparty is an external-bank transfer, as in the source schema.
    linked = tx.loc[external, ["transfer_id", "occurrence", "direction"]]
    paired = linked.groupby(["transfer_id", "occurrence"]).direction.transform("nunique").eq(2)
    tx.loc[external, "transfer_id"] = ("P" + linked.occurrence.astype(str) + "-" + linked.transfer_id).where(paired)
    scaled = (tx.amount * tx.money_scale).round(2).clip(lower=.01)
    tx["amount"] = tx.amount.where(external, scaled)
    cash = tx.transaction_type.isin(("cash_deposit", "cash_withdrawal"))
    tx.loc[cash, "amount"] = (tx.loc[cash, "amount"] / 10).round().clip(lower=1) * 10

    savings = account_rows("savings_cycles")
    savings["init_balance"] = (savings.init_balance * savings.money_scale).round(2).clip(lower=.01)
    login, complaints = customer_rows("login"), customer_rows("complaints")
    login["login_id"] = scoped_id(login.customer_id, login.login_id)
    complaints["complaint_id"] = scoped_id(complaints.customer_id, complaints.complaint_id)

    result = {
        "kaggle_bank_churn": base.reset_index(), "customer_context": customer_rows("customer_context"),
        "accounts": accounts, "savings_cycles": savings, "transactions": tx, "login": login,
        "complaints": complaints,
    }
    for name, frame in result.items():
        result[name] = frame.loc[:, tables[name].columns].copy()
        for column in DATES.get(name, ()):
            result[name][column] = warp_clock(frame[column], warp_seconds)
    return result
