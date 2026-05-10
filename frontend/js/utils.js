export function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  const formatted = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 }).format(value);
  return `${formatted} DH`;
}

export function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 }).format(value);
}

export function scoreClass(score) {
  if (score >= 8) return "excellent";
  if (score >= 6.5) return "good";
  if (score >= 4) return "mid";
  return "low";
}

export function stars(score) {
  const rounded = Math.round(score);
  return "★".repeat(rounded) + "☆".repeat(Math.max(0, 5 - rounded));
}

export function showSkeleton(container, lines = 3) {
  container.innerHTML = Array.from({ length: lines })
    .map(() => '<div class="skeleton" style="height:18px;margin-bottom:10px;"></div>')
    .join("");
}

export async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}
