export function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const formatted = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 }).format(value);
  return `${formatted} DH`;
}

export function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 }).format(value);
}

// Couleurs strictement liées à ton fichier main.css
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
    .map(() => '<div class="skeleton" style="height:20px;margin-bottom:12px;"></div>')
    .join("");
}

const API_BASE_URL = window.__API_BASE_URL__ || "";

function resolveUrl(url) {
  if (url.startsWith("http")) return url;
  if (url.startsWith("/")) return `${API_BASE_URL}${url}`;
  return `${API_BASE_URL}/${url}`;
}

export async function fetchJSON(url, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000);

  try {
    const response = await fetch(resolveUrl(url), { ...options, signal: controller.signal });
    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`Request failed: ${response.status} ${errorBody}`);
    }
    return response.json();
  } finally {
    clearTimeout(timeoutId);
  }
}