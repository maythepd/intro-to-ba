import type { CustomerFeatures } from "./api";

/**
 * DEMO FIXTURES — hand-written records for UI preview only.
 * They are NOT real customers and are NOT drawn from the bank dataset.
 * Every place that renders them must show a "Demo" stamp.
 */
export const DEMO_CUSTOMERS: CustomerFeatures[] = [
  { customer_id: "DEMO-0001", credit_score: 648, country: "France", gender: "Female", age: 42, tenure: 6, balance: 0, products_number: 2, credit_card: 1, active_member: 1, estimated_salary: 101200, recency_days: 4, monthly_transactions: 31, trans_amount_3m: 8420, trans_trend_ratio: 1.04, app_logins_monthly: 22, customer_service_complaints: 0, has_unresolved_complaint: 0, preferred_channel: "Mobile App", primary_device: "iOS" },
  { customer_id: "DEMO-0002", credit_score: 590, country: "Germany", gender: "Male", age: 51, tenure: 2, balance: 118400, products_number: 1, credit_card: 1, active_member: 0, estimated_salary: 74300, recency_days: 46, monthly_transactions: 6, trans_amount_3m: 1210, trans_trend_ratio: 0.48, app_logins_monthly: 2, customer_service_complaints: 3, has_unresolved_complaint: 1, preferred_channel: "Branch", primary_device: "Web/Desktop" },
  { customer_id: "DEMO-0003", credit_score: 712, country: "Spain", gender: "Female", age: 34, tenure: 8, balance: 54200, products_number: 2, credit_card: 0, active_member: 1, estimated_salary: 58900, recency_days: 2, monthly_transactions: 40, trans_amount_3m: 6900, trans_trend_ratio: 1.12, app_logins_monthly: 28, customer_service_complaints: 0, has_unresolved_complaint: 0, preferred_channel: "Mobile App", primary_device: "Android" },
  { customer_id: "DEMO-0004", credit_score: 535, country: "Germany", gender: "Female", age: 47, tenure: 1, balance: 97600, products_number: 3, credit_card: 1, active_member: 0, estimated_salary: 132000, recency_days: 63, monthly_transactions: 3, trans_amount_3m: 540, trans_trend_ratio: 0.31, app_logins_monthly: 1, customer_service_complaints: 2, has_unresolved_complaint: 1, preferred_channel: "Online Web", primary_device: "Web/Desktop" },
  { customer_id: "DEMO-0005", credit_score: 680, country: "France", gender: "Male", age: 29, tenure: 4, balance: 21500, products_number: 2, credit_card: 1, active_member: 1, estimated_salary: 46800, recency_days: 7, monthly_transactions: 24, trans_amount_3m: 3980, trans_trend_ratio: 0.92, app_logins_monthly: 15, customer_service_complaints: 1, has_unresolved_complaint: 0, preferred_channel: "Mobile App", primary_device: "Android" },
  { customer_id: "DEMO-0006", credit_score: 621, country: "Spain", gender: "Male", age: 58, tenure: 9, balance: 143000, products_number: 1, credit_card: 0, active_member: 0, estimated_salary: 88100, recency_days: 28, monthly_transactions: 9, trans_amount_3m: 2150, trans_trend_ratio: 0.66, app_logins_monthly: 4, customer_service_complaints: 1, has_unresolved_complaint: 0, preferred_channel: "ATM", primary_device: "iOS" },
  { customer_id: "DEMO-0007", credit_score: 755, country: "France", gender: "Female", age: 38, tenure: 7, balance: 76300, products_number: 2, credit_card: 1, active_member: 1, estimated_salary: 112500, recency_days: 1, monthly_transactions: 36, trans_amount_3m: 9120, trans_trend_ratio: 1.08, app_logins_monthly: 25, customer_service_complaints: 0, has_unresolved_complaint: 0, preferred_channel: "Online Web", primary_device: "iOS" },
  { customer_id: "DEMO-0008", credit_score: 602, country: "Germany", gender: "Male", age: 44, tenure: 3, balance: 109800, products_number: 1, credit_card: 1, active_member: 1, estimated_salary: 67400, recency_days: 19, monthly_transactions: 12, trans_amount_3m: 2610, trans_trend_ratio: 0.74, app_logins_monthly: 7, customer_service_complaints: 2, has_unresolved_complaint: 0, preferred_channel: "Branch", primary_device: "Android" },
];

export function findDemoCustomer(id: string) {
  return DEMO_CUSTOMERS.find((c) => c.customer_id.toLowerCase() === id.toLowerCase());
}

export function filterDemoCustomers(search: string) {
  const q = search.trim().toLowerCase();
  if (!q) return DEMO_CUSTOMERS;
  return DEMO_CUSTOMERS.filter((c) =>
    [c.customer_id, c.country, c.gender, c.preferred_channel, c.primary_device].some((v) => v.toLowerCase().includes(q)),
  );
}

/** Source-reported figures (EDA report / source study). Not deployment measurements. */
export const SOURCE_FACTS = {
  totalCustomers: 10000,
  baselineChurnRate: 0.2037,
  xgboost: { rocAuc: 0.9999, prAuc: 0.9997, f1: 0.9914, recall: 0.9877 },
} as const;