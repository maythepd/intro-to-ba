export const fmtInt = (n: number) => new Intl.NumberFormat("en-US").format(n);
export const fmtMoney = (n: number) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(n);
export const fmtPct = (n: number, digits = 2) => `${(n * 100).toFixed(digits)}%`;
export const yesNo = (v: number | boolean) => (v ? "Yes" : "No");