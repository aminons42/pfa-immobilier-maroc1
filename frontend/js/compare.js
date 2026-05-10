import { fetchJSON, formatMoney, formatNumber, showSkeleton } from "./utils.js";

const cityA = document.getElementById("cityA");
const cityB = document.getElementById("cityB");
const typeFilter = document.getElementById("typeFilter");
const compareBtn = document.getElementById("compareBtn");
const compareTable = document.getElementById("compareTable");
const verdictCard = document.getElementById("verdictCard");

let trendChart = null;

async function loadCities() {
  const cities = await fetchJSON("/api/cities");
  cityA.innerHTML = cities.map((city) => `<option>${city.name}</option>`).join("");
  cityB.innerHTML = cities.map((city) => `<option>${city.name}</option>`).join("");
  if (cities[1]) cityB.value = cities[1].name;
}

function row(label, a, b, highlight) {
  return `
    <tr>
      <th>${label}</th>
      <td class="${highlight === "a" ? "badge" : ""}">${a}</td>
      <td class="${highlight === "b" ? "badge" : ""}">${b}</td>
    </tr>
  `;
}

function highlight(a, b, higher = true) {
  if (higher) return a >= b ? "a" : "b";
  return a <= b ? "a" : "b";
}

async function compare() {
  compareTable.innerHTML = "";
  verdictCard.innerHTML = "";
  showSkeleton(compareTable, 6);

  const payload = {
    city_a: cityA.value,
    city_b: cityB.value,
    type_bien: typeFilter.value || null,
  };

  const result = await fetchJSON("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const a = result.city_a;
  const b = result.city_b;

  compareTable.innerHTML = `
    <table class="table">
      <thead>
        <tr><th>Indicateur</th><th>${a.city}</th><th>${b.city}</th></tr>
      </thead>
      <tbody>
        ${row("Prix/m² moyen", formatMoney(a.avg_prix_m2), formatMoney(b.avg_prix_m2), highlight(a.avg_prix_m2, b.avg_prix_m2, false))}
        ${row("Prix total moyen", formatMoney(a.avg_prix_total), formatMoney(b.avg_prix_total), highlight(a.avg_prix_total, b.avg_prix_total, false))}
        ${row("Surface moyenne", `${formatNumber(a.avg_surface)} m²`, `${formatNumber(b.avg_surface)} m²`, highlight(a.avg_surface, b.avg_surface, true))}
        ${row("Nombre d'annonces", formatNumber(a.listing_count), formatNumber(b.listing_count), highlight(a.listing_count, b.listing_count, true))}
        ${row("Score d'investissement", a.investment_score.toFixed(1), b.investment_score.toFixed(1), highlight(a.investment_score, b.investment_score, true))}
        ${row("Croissance annuelle estimée", `${(a.annual_growth_rate * 100).toFixed(1)}%`, `${(b.annual_growth_rate * 100).toFixed(1)}%`, highlight(a.annual_growth_rate, b.annual_growth_rate, true))}
      </tbody>
    </table>
  `;

  const winnerGrowth = result.verdict.long_term;
  const winnerBudget = result.verdict.budget;

  verdictCard.innerHTML = `
    <h3>🏆 VERDICT</h3>
    <p>Pour un investissement long terme : <strong>${winnerGrowth}</strong> est recommandé (croissance ${result.verdict.growth_rate.toFixed(2)})</p>
    <p>Pour un budget accessible : <strong>${winnerBudget}</strong> offre plus de choix (${formatNumber(result.verdict.listing_count)} annonces)</p>
  `;

  await loadTrendComparison(a.city, b.city);
}

async function loadTrendComparison(cityOne, cityTwo) {
  const trendA = await fetchJSON(`/api/trend/${cityOne}`);
  const trendB = await fetchJSON(`/api/trend/${cityTwo}`);

  const labels = trendA.historical.map((point) => `${point.year}-${String(point.month).padStart(2, "0")}`);

  const ctx = document.getElementById("compareTrend");
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: cityOne,
          data: trendA.historical.map((p) => p.value),
          borderColor: "#C9A84C",
          borderWidth: 2,
          pointRadius: 0,
        },
        {
          label: cityTwo,
          data: trendB.historical.map((p) => p.value),
          borderColor: "#f5f5f0",
          borderWidth: 2,
          pointRadius: 0,
          borderDash: [6, 6],
        },
      ],
    },
    options: {
      plugins: { legend: { labels: { color: "#f5f5f0" } } },
      scales: { x: { ticks: { color: "#8899AA", maxTicksLimit: 6 } }, y: { ticks: { color: "#8899AA" } } },
    },
  });
}

compareBtn.addEventListener("click", compare);
loadCities();
