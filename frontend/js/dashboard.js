import { fetchJSON, formatMoney, formatNumber, showSkeleton } from "./utils.js";

const cityPills = document.getElementById("cityPills");
const statsGrid = document.getElementById("statsGrid");

let typeChart = null;
let quartierChart = null;

async function loadCities() {
  try {
    const cities = await fetchJSON("/api/cities");
    cityPills.innerHTML = cities.map((city, index) => `<div class="pill ${index === 0 ? "active" : ""}" data-city="${city.name}">${city.name}</div>`).join("");
    cityPills.querySelectorAll(".pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        cityPills.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        loadStats(pill.dataset.city);
      });
    });
    if (cities[0]) loadStats(cities[0].name);
  } catch (error) {
    cityPills.innerHTML = "<div class=\"text-slate-400\">Chargement impossible</div>";
  }
}

async function loadStats(city) {
  statsGrid.innerHTML = "";
  showSkeleton(statsGrid, 4);
  try {
    const stats = await fetchJSON(`/api/stats/${encodeURIComponent(city)}`);

    statsGrid.innerHTML = `
      <div class="card">Volume d'annonces<br/><strong style="font-size:1.4rem; color:var(--text);">${formatNumber(stats.listing_count)}</strong></div>
      <div class="card">Prix / m² Moyen<br/><strong style="font-size:1.4rem; color:var(--accent);">${formatMoney(stats.avg_prix_m2)}</strong></div>
      <div class="card">Prix Median du Marche<br/><strong style="font-size:1.4rem; color:var(--text);">${formatMoney(stats.price_range.median)}</strong></div>
      <div class="card">Surface Moyenne<br/><strong style="font-size:1.4rem; color:var(--text);">${formatNumber(stats.avg_surface)} m²</strong></div>
    `;

    buildTypeChart(stats.type_distribution);
    buildQuartierChart(stats.top_quartiers);
  } catch (error) {
    statsGrid.innerHTML = "<div class=\"text-slate-400\">Aucune donnee disponible</div>";
  }
}

function buildTypeChart(items) {
  const ctx = document.getElementById("typeChart");
  if (typeChart) typeChart.destroy();
  typeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: items.map((item) => item.type),
      datasets: [{ data: items.map((item) => item.count), backgroundColor: ["#C9A84C", "#b48d39", "#e0c26b"], borderWidth: 0 }],
    },
    options: { plugins: { legend: { labels: { color: "#1f2937", font: { family: "Space Grotesk" } } } } },
  });
}

function buildQuartierChart(items) {
  const ctx = document.getElementById("quartierChart");
  if (quartierChart) quartierChart.destroy();
  quartierChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: items.map((item) => item.name),
      datasets: [{ data: items.map((item) => item.avg_prix_m2), backgroundColor: "#C9A84C", borderRadius: 4 }],
    },
    options: {
      indexAxis: "y",
      plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false }, ticks: { color: "#64748b" } }, y: { grid: { display: false }, ticks: { color: "#1f2937", font: { weight: "bold" } } } },
    },
  });
}

loadCities();