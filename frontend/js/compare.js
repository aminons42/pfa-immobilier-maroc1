import { fetchJSON, formatMoney, formatNumber, showSkeleton } from "./utils.js";

const cityA = document.getElementById("cityA");
const cityB = document.getElementById("cityB");
const typeFilter = document.getElementById("typeFilter");
const compareBtn = document.getElementById("compareBtn");
const compareTable = document.getElementById("compareTable");
const verdictCard = document.getElementById("verdictCard");
const trendSummary = document.getElementById("trendSummary");
const trendBars = document.getElementById("trendBars");

async function loadCities() {
  try {
    const cities = await fetchJSON("/api/cities");
    cityA.innerHTML = cities.map((city) => `<option value="${city.name}">${city.name}</option>`).join("");
    cityB.innerHTML = cities.map((city) => `<option value="${city.name}">${city.name}</option>`).join("");
    if (cities[1]) cityB.value = cities[1].name;
  } catch (error) {
    compareTable.innerHTML = "<div class=\"text-slate-400\">Chargement impossible</div>";
  }
}

function row(label, a, b, highlight) {
  return `
    <tr>
      <th style="color:var(--muted); font-weight:500;">${label}</th>
      <td class="${highlight === "a" ? "badge" : ""}" style="text-align:right;">${a}</td>
      <td class="${highlight === "b" ? "badge" : ""}" style="text-align:right;">${b}</td>
    </tr>
  `;
}

function highlight(a, b, higher = true) {
  if (higher) return a >= b ? "a" : "b";
  return a <= b ? "a" : "b";
}

function percentDiff(a, b) {
  if (!a || !b) return 0;
  return Math.round(((a - b) / b) * 100);
}

function renderTrends(a, b) {
  const priceDelta = percentDiff(a.avg_prix_m2, b.avg_prix_m2);
  const volumeDelta = percentDiff(a.listing_count, b.listing_count);
  const surfaceDelta = percentDiff(a.avg_surface, b.avg_surface);

  trendSummary.textContent = `Prix/m² ${priceDelta >= 0 ? "plus élevé" : "plus bas"} à ${a.city} (${Math.abs(priceDelta)}%), volume ${volumeDelta >= 0 ? "plus élevé" : "plus bas"} (${Math.abs(volumeDelta)}%), surface ${surfaceDelta >= 0 ? "plus grande" : "plus petite"} (${Math.abs(surfaceDelta)}%).`;

  const priceMax = Math.max(a.avg_prix_m2, b.avg_prix_m2, 1);
  const volumeMax = Math.max(a.listing_count, b.listing_count, 1);
  const surfaceMax = Math.max(a.avg_surface, b.avg_surface, 1);

  trendBars.innerHTML = `
    <div class="trend-bar">
      <span>Prix/m²</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(a.avg_prix_m2 / priceMax) * 100}%">${a.city}</div>
        <div class="bar-fill alt" style="width:${(b.avg_prix_m2 / priceMax) * 100}%">${b.city}</div>
      </div>
    </div>
    <div class="trend-bar">
      <span>Volume</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(a.listing_count / volumeMax) * 100}%">${a.city}</div>
        <div class="bar-fill alt" style="width:${(b.listing_count / volumeMax) * 100}%">${b.city}</div>
      </div>
    </div>
    <div class="trend-bar">
      <span>Surface</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(a.avg_surface / surfaceMax) * 100}%">${a.city}</div>
        <div class="bar-fill alt" style="width:${(b.avg_surface / surfaceMax) * 100}%">${b.city}</div>
      </div>
    </div>
  `;
}

function renderVerdict(a, b) {
  const priceDiff = a.avg_prix_m2 - b.avg_prix_m2;
  const priceWinner = priceDiff >= 0 ? a.city : b.city;
  const volumeWinner = a.listing_count >= b.listing_count ? a.city : b.city;
  const surfaceWinner = a.avg_surface >= b.avg_surface ? a.city : b.city;

  verdictCard.innerHTML = `
    <div class="verdict-title">Verdict IA</div>
    <div class="verdict-body">
      <p><strong>${priceWinner}</strong> est plus cher au m².</p>
      <p><strong>${volumeWinner}</strong> a le plus d'annonces actives.</p>
      <p><strong>${surfaceWinner}</strong> offre les plus grandes surfaces.</p>
    </div>
  `;
}

async function compare() {
  compareTable.innerHTML = "";
  showSkeleton(compareTable, 4);

  try {
    const payload = { city_a: cityA.value, city_b: cityB.value, type_bien: typeFilter.value || null };
    const result = await fetchJSON("/api/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

    const a = result.city_a;
    const b = result.city_b;

    if (!a || !b || (a.listing_count === 0 && b.listing_count === 0)) {
      trendSummary.textContent = "Aucune donnee disponible pour cette comparaison.";
      trendBars.innerHTML = "";
      verdictCard.innerHTML = "<div class=\"text-center text-indigo-300 py-4\">Le verdict de l'IA s'affichera ici</div>";
      return;
    }

    compareTable.innerHTML = `
      <table class="table">
        <thead>
          <tr><th style="color:var(--accent);">Donnees reelles du dataset</th><th style="text-align:right;">${a.city}</th><th style="text-align:right;">${b.city}</th></tr>
        </thead>
        <tbody>
          ${row("Prix/m2 Moyen", formatMoney(a.avg_prix_m2), formatMoney(b.avg_prix_m2), highlight(a.avg_prix_m2, b.avg_prix_m2, false))}
          ${row("Capital Moyen du Bien", formatMoney(a.avg_prix_total), formatMoney(b.avg_prix_total), highlight(a.avg_prix_total, b.avg_prix_total, false))}
          ${row("Surface Moyenne", `${formatNumber(a.avg_surface)} m2`, `${formatNumber(b.avg_surface)} m2`, highlight(a.avg_surface, b.avg_surface, true))}
          ${row("Volume d'annonces", formatNumber(a.listing_count), formatNumber(b.listing_count), highlight(a.listing_count, b.listing_count, true))}
        </tbody>
      </table>
    `;

    renderTrends(a, b);
    renderVerdict(a, b);
  } catch (error) {
    compareTable.innerHTML = "<div class=\"text-slate-400\">Erreur de comparaison</div>";
  }
}

compareBtn.addEventListener("click", compare);
loadCities();