/** Committed Module 5 holdout/CV figures, not deployment measurements. */
export const MODEL_BENCHMARKS = [
  { model: "Logistic Regression", rocAuc: 0.5985, prAuc: 0.2590, f1: 0.3570, precision: 0.2307, recall: 0.7892, accuracy: 0.4200 },
  { model: "Random Forest", rocAuc: 0.8213, prAuc: 0.5760, f1: 0.5469, precision: 0.5833, recall: 0.5147, accuracy: 0.8260 },
  { model: "XGBoost", rocAuc: 0.8372, prAuc: 0.6060, f1: 0.3389, precision: 0.2040, recall: 1.0000, accuracy: 0.2040 },
] as const;

export const SOURCE_FACTS = {
  totalCustomers: 10000,
  baselineChurnRate: 0.2037,
  xgboost: { rocAuc: 0.8372, prAuc: 0.6060, f1: 0.3389, recall: 1.0000 },
} as const;
