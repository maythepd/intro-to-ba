/** Source-reported figures (EDA report / source study). Not deployment measurements. */
export const SOURCE_FACTS = {
  totalCustomers: 10000,
  baselineChurnRate: 0.2037,
  xgboost: { rocAuc: 0.9999, prAuc: 0.9997, f1: 0.9914, recall: 0.9877 },
} as const;