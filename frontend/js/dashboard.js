import { fetchJSON, formatMoney, formatNumber, scoreClass, stars, showSkeleton } from "./utils.js";

const cityPills = document.getElementById("cityPills");
const statsGrid = document.getElementById("statsGrid");

let typeChart = null;
let quartierChart = null;
let rangeChart = null;
let trendChart = null;

async function loadCities() {
  const cities = await fetchJSON("/api/cities");
  cityPills.innerHTML = cities.map((city, index) => `<div class="pill ${index === 0 ? "active" : ""}" data-city="${city.name}">${city.name}</div>`).join("");
  cityPills.querySelectorAll(".pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      cityPills.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      loadStats(pill.dataset.city);
    });
  });
  if (cities[0]) {
    loadStats(cities[0].name);
  }
}

async function loadStats(city) {
  statsGrid.innerHTML = "";
  showSkeleton(statsGrid, 6);
  const stats = await fetchJSON(`/api/stats/${city}`);

  statsGrid.innerHTML = `
    <div class="card">Total annonces<br/><strong>${formatNumber(stats.listing_count)}</strong></div>
    <div class="card">Prix/m² moyen<br/><strong>${formatMoney(stats.avg_prix_m2)}</strong></div>
    <div class="card">Prix total médian<br/><strong>${formatMoney(stats.price_range.median)}</strong></div>
    <div class="card">Surface moyenne<br/><strong>${formatNumber(stats.avg_surface)} m²</strong></div>
    <div class="card">Score d'investissement<br/><strong class="score ${scoreClass(stats.investment_score)}">${stars(stats.investment_score)} ${stats.investment_score.toFixed(1)}</strong></div>
    <div class="card">Quartier le plus cher<br/><strong>${stats.top_quartiers[0]?.name || "-"} (${formatMoney(stats.top_quartiers[0]?.avg_prix_m2 || 0)})</strong></div>
  `;

  buildTypeChart(stats.type_distribution);
  buildQuartierChart(stats.top_quartiers);
  buildRangeChart(stats.price_range);
  buildTrendChart(stats.trend);
}

function buildTypeChart(items) {
  const ctx = document.getElementById("typeChart");
  if (typeChart) typeChart.destroy();
  typeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: items.map((item) => item.type),
      datasets: [{
        data: items.map((item) => item.count),
        backgroundColor: ["#C9A84C", "#b48d39", "#e0c26b", "#a0792c", "#f0d28a"],
      }],
    },
    options: { plugins: { legend: { labels: { color: "#f5f5f0" } } } },
  });
}

function buildQuartierChart(items) {
  const ctx = document.getElementById("quartierChart");
  if (quartierChart) quartierChart.destroy();
  quartierChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: items.map((item) => item.name),
      datasets: [{
        data: items.map((item) => item.avg_prix_m2),
        backgroundColor: "#C9A84C",
      }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#8899AA" } }, y: { ticks: { color: "#8899AA" } } },
    },
  });
}

function buildRangeChart(range) {
  const ctx = document.getElementById("rangeChart");
  if (rangeChart) rangeChart.destroy();

  rangeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Range"],
      datasets: [{
        data: [range.max - range.min],
        backgroundColor: "rgba(201,168,76,0.2)",
        borderColor: "#C9A84C",
        borderWidth: 1,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { display: false } },
    },
  });
}

function buildTrendChart(points) {
  const ctx = document.getElementById("trendChart");
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: points.map((p) => `${p.year}-${String(p.month).padStart(2, "0")}`),
      datasets: [{
        data: points.map((p) => p.value),
        borderColor: "#C9A84C",
        borderWidth: 2,
        pointRadius: 0,
      }],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { x: { ticks: { color: "#8899AA", maxTicksLimit: 6 } }, y: { ticks: { color: "#8899AA" } } },
    },
  });
}

loadCities();
