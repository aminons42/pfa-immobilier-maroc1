import { fetchJSON, formatMoney, formatNumber, scoreClass, stars, showSkeleton } from "./utils.js";

const villeSelect = document.getElementById("ville");
const quartierSelect = document.getElementById("quartier");
const form = document.getElementById("estimatorForm");
const resultsContent = document.getElementById("resultsContent");
const trendSection = document.getElementById("trendSection");
const trendMeta = document.getElementById("trendMeta");
const trendChart = document.getElementById("trendChart");

let trendChartInstance = null;

function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

async function loadCities() {
  const cities = await fetchJSON("/api/cities");
  villeSelect.innerHTML = cities.map((city) => `<option>${city.name}</option>`).join("");
  const preselect = getQueryParam("ville");
  if (preselect) {
    villeSelect.value = preselect;
  }
  await loadQuartiers();
}

async function loadQuartiers() {
  quartierSelect.disabled = true;
  quartierSelect.innerHTML = "<option>Chargement...</option>";
  const city = villeSelect.value;
  const quartiers = await fetchJSON(`/api/cities/${city}/quartiers`);
  quartierSelect.innerHTML = quartiers.map((q) => `<option>${q.name}</option>`).join("");
  quartierSelect.innerHTML += `<option>Autre secteur</option>`;
  const preselect = getQueryParam("quartier");
  if (preselect) {
    quartierSelect.value = preselect;
  }
  quartierSelect.disabled = false;
}

villeSelect.addEventListener("change", loadQuartiers);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultsContent.innerHTML = "";
  showSkeleton(resultsContent, 6);

  const payload = {
    ville: villeSelect.value,
    quartier: quartierSelect.value,
    type_bien: document.getElementById("typeBien").value,
    surface: Number(document.getElementById("surface").value),
    nb_chambres: Number(document.getElementById("chambres").value),
    nb_salles_bain: Number(document.getElementById("sdb").value),
    etat_bien: document.getElementById("etat").value,
    etage: Number(document.getElementById("etage").value),
  };

  const result = await fetchJSON("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const rangePosition = Math.min(100, Math.max(0, ((result.estimation - result.range_min) / Math.max(result.range_max - result.range_min, 1)) * 100));

  resultsContent.innerHTML = `
    <h2>ESTIMATION DU PRIX</h2>
    <div class="result-range">
      <span>${formatMoney(result.range_min)}</span>
      <div class="range-bar"><span class="range-dot" style="left:${rangePosition}%;"></span></div>
      <span>${formatMoney(result.range_max)}</span>
    </div>
    <p style="margin:12px 0;">Estimation : <strong>${formatMoney(result.estimation)}</strong></p>
    <hr style="border-color:rgba(201,168,76,0.2);" />
    <p>Prix au m² estimé : <strong>${formatMoney(result.prix_m2_estime)}</strong></p>
    <p>Intervalle de confiance : ${result.confidence}</p>
    <hr style="border-color:rgba(201,168,76,0.2);" />
    <h3>CONTEXTE MARCHÉ</h3>
    <p>Moyenne ${payload.ville} : ${formatMoney(result.market_context.avg_prix_m2_city)}</p>
    <p>Moyenne ${payload.quartier} : ${formatMoney(result.market_context.avg_prix_m2_quartier)}</p>
    <p>${result.market_context.position === "above_market" ? "▲" : "▼"} Votre bien est ${result.market_context.position === "above_market" ? "AU-DESSUS" : "EN-DESSOUS"} du marché</p>
    <p>Percentile : ${result.market_context.percentile}e parmi les biens similaires</p>
    <hr style="border-color:rgba(201,168,76,0.2);" />
    <h3>SCORE D'INVESTISSEMENT</h3>
    <p class="score ${scoreClass(result.investment_score)}">${stars(result.investment_score)} ${result.investment_score} / 10</p>
    <p>"${result.investment_verdict} — marché en croissance"</p>
    <button class="btn btn-outline" id="trendBtn">Voir la tendance des prix →</button>
  `;

  document.getElementById("trendBtn").addEventListener("click", () => loadTrend(payload, result));
});

async function loadTrend(payload, result) {
  trendSection.style.display = "block";
  showSkeleton(trendMeta, 3);
  const trend = await fetchJSON(`/api/trend/${payload.ville}?type_bien=${encodeURIComponent(payload.type_bien)}`);
  const historical = trend.historical || [];
  const forecast = trend.forecast || [];
  const labels = [...historical, ...forecast].map((point) => `${point.year}-${String(point.month).padStart(2, "0")}`);
  const values = [...historical, ...forecast].map((point) => point.value);
  const lows = [...historical, ...forecast].map((point) => point.low);
  const highs = [...historical, ...forecast].map((point) => point.high);
  const todayIndex = historical.length - 1;

  const plugin = {
    id: "todayLine",
    afterDraw(chart) {
      const { ctx, chartArea } = chart;
      if (!chartArea || todayIndex < 0) return;
      const x = chart.scales.x.getPixelForValue(todayIndex);
      ctx.save();
      ctx.strokeStyle = "#f5f5f0";
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      ctx.restore();
    },
  };

  const chartData = {
    labels,
    datasets: [
      {
        label: "Prix m²",
        data: values,
        borderColor: "#C9A84C",
        backgroundColor: "rgba(201,168,76,0.15)",
        borderWidth: 2,
        pointRadius: 0,
      },
      {
        label: "Low",
        data: lows,
        borderColor: "transparent",
        backgroundColor: "rgba(201,168,76,0.1)",
        fill: "+1",
        pointRadius: 0,
      },
      {
        label: "High",
        data: highs,
        borderColor: "transparent",
        backgroundColor: "rgba(201,168,76,0.1)",
        fill: false,
        pointRadius: 0,
      },
    ],
  };

  if (trendChartInstance) {
    trendChartInstance.destroy();
  }

  trendChartInstance = new Chart(trendChart, {
    type: "line",
    data: chartData,
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: {
            callback: (value, index) => labels[index].split("-")[0],
            color: "#8899AA",
            maxTicksLimit: 8,
          },
        },
        y: { ticks: { color: "#8899AA" } },
      },
    },
    plugins: [plugin],
  });

  const growth = trend.annual_growth_rate * 100;
  const projected = result.estimation * (1 + trend.annual_growth_rate * 10);
  const projectionPercent = trend.annual_growth_rate * 10 * 100;

  trendMeta.innerHTML = `
    <p>Croissance annuelle estimée : +${growth.toFixed(1)}%/an</p>
    <p>Si vous achetez aujourd'hui à ${formatMoney(result.estimation)}, valeur estimée dans 10 ans : ${formatMoney(projected)} (+${projectionPercent.toFixed(0)}%)</p>
  `;
}

loadCities();
