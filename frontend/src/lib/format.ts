export const fmtInt = (n: number) => new Intl.NumberFormat("vi-VN").format(n);
export const fmtMoney = (n: number) =>
  new Intl.NumberFormat("vi-VN", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(n);
export const fmtPct = (n: number, digits = 2) => `${(n * 100).toFixed(digits).replace(".", ",")}%`;
export const yesNo = (v: number | boolean) => (v ? "Có" : "Không");