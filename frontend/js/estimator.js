import { fetchJSON, formatMoney, formatNumber, showSkeleton } from "./utils.js";

const villeSelect = document.getElementById("ville");
const quartierSelect = document.getElementById("quartier");
const form = document.getElementById("estimatorForm");
const resultsContent = document.getElementById("resultsContent");

function getQueryParam(name) {
  const params = new URLSearchParams(window.location.search);
  return params.get(name);
}

async function loadCities() {
  try {
    const cities = await fetchJSON("/api/cities");
    villeSelect.innerHTML = cities
      .map((city) => `<option value="${city.name}">${city.name}</option>`)
      .join("");
    const preselect = getQueryParam("ville");
    if (preselect) villeSelect.value = preselect;
    await loadQuartiers();
  } catch (error) {
    villeSelect.innerHTML = "<option>Erreur de chargement</option>";
    quartierSelect.innerHTML = "<option>Erreur de chargement</option>";
  }
}

async function loadQuartiers() {
  quartierSelect.disabled = true;
  quartierSelect.innerHTML = "<option>Chargement...</option>";
  const city = villeSelect.value;
  if (!city) return;
  try {
    const quartiers = await fetchJSON(
      `/api/cities/${encodeURIComponent(city)}/quartiers`
    );
    quartierSelect.innerHTML = quartiers
      .map((q) => `<option value="${q.name}">${q.name}</option>`)
      .join("");
    const preselect = getQueryParam("quartier");
    if (preselect) quartierSelect.value = preselect;
  } catch (error) {
    quartierSelect.innerHTML = "<option>Erreur de chargement</option>";
  } finally {
    quartierSelect.disabled = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// NOUVELLE FONCTION — Affiche le graphique de tendance des prix sur 5 ans
// Elle est appelée automatiquement après l'affichage de l'estimation.
// ─────────────────────────────────────────────────────────────────────────────
function renderTrendChart(forecast, taux_annuel, variation_5ans) {
  // Prépare les données pour Chart.js
  const labels = forecast.map((p) => p.year.toString());
  const prixData = forecast.map((p) => Math.round(p.prix));
  const prixM2Data = forecast.map((p) => Math.round(p.prix_m2));

  // Couleur du badge : vert si hausse, rouge si baisse
  const isPositive = variation_5ans >= 0;
  const badgeColor = isPositive ? "#16a34a" : "#dc2626";
  const arrow = isPositive ? "↑" : "↓";
  const variationText = `${arrow} ${Math.abs(variation_5ans).toFixed(1)}% sur 5 ans`;

  // Crée le bloc HTML de la section tendance et l'injecte sous l'estimation
  const trendSection = document.createElement("div");
  trendSection.style.cssText =
    "margin-top:28px; border-top:1px solid var(--border); padding-top:20px;";
  trendSection.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
      <h3 style="font-size:0.85rem; font-weight:700; text-transform:uppercase;
                 letter-spacing:0.08em; color:var(--accent); margin:0;">
        Tendance sur 5 ans
      </h3>
      <div style="display:flex; gap:8px; align-items:center;">
        <span style="background:${badgeColor}18; color:${badgeColor}; font-size:0.78rem;
                     font-weight:700; padding:3px 10px; border-radius:999px; border:1px solid ${badgeColor}33;">
          ${variationText}
        </span>
        <span style="color:var(--muted); font-size:0.78rem;">
          ${taux_annuel.toFixed(1)}%/an
        </span>
      </div>
    </div>

    <canvas id="trendChart" height="140"></canvas>

    <p style="font-size:0.72rem; color:var(--muted); margin-top:10px; text-align:center; line-height:1.5;">
      Projection basée sur les tendances du marché immobilier marocain par ville.
      À titre indicatif — les conditions de marché peuvent évoluer.
    </p>
  `;
  resultsContent.appendChild(trendSection);

  // Dessine le graphique avec Chart.js (déjà chargé dans estimator.html)
  const ctx = document.getElementById("trendChart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Prix total (DH)",
          data: prixData,
          borderColor: "#5b5bd6",
          backgroundColor: "rgba(91,91,214,0.08)",
          borderWidth: 2.5,
          pointBackgroundColor: "#5b5bd6",
          pointRadius: 5,
          pointHoverRadius: 7,
          fill: true,
          tension: 0.35,
          yAxisID: "yPrix",
        },
        {
          label: "Prix/m² (DH)",
          data: prixM2Data,
          borderColor: "#f4c75b",
          backgroundColor: "rgba(244,199,91,0)",
          borderWidth: 2,
          pointBackgroundColor: "#f4c75b",
          pointRadius: 4,
          pointHoverRadius: 6,
          borderDash: [5, 3],
          fill: false,
          tension: 0.35,
          yAxisID: "yM2",
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "bottom",
          labels: { font: { size: 11 }, boxWidth: 12, padding: 16 },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.y;
              const formatted = new Intl.NumberFormat("fr-FR", {
                maximumFractionDigits: 0,
              }).format(val);
              return `${ctx.dataset.label}: ${formatted} DH`;
            },
          },
        },
      },
      scales: {
        yPrix: {
          type: "linear",
          position: "left",
          ticks: {
            font: { size: 10 },
            callback: (v) =>
              new Intl.NumberFormat("fr-FR", {
                notation: "compact",
                maximumFractionDigits: 1,
              }).format(v) + " DH",
          },
          grid: { color: "rgba(0,0,0,0.06)" },
        },
        yM2: {
          type: "linear",
          position: "right",
          ticks: {
            font: { size: 10 },
            callback: (v) =>
              new Intl.NumberFormat("fr-FR", {
                maximumFractionDigits: 0,
              }).format(v) + " /m²",
          },
          grid: { drawOnChartArea: false },
        },
        x: {
          ticks: { font: { size: 11 } },
          grid: { color: "rgba(0,0,0,0.04)" },
        },
      },
    },
  });
}
// ─────────────────────────────────────────────────────────────────────────────

villeSelect.addEventListener("change", loadQuartiers);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultsContent.innerHTML = "";
  showSkeleton(resultsContent, 4);

  const payload = {
    ville: villeSelect.value,
    quartier: quartierSelect.value,
    type_bien: document.getElementById("typeBien").value,
    surface: Number(document.getElementById("surface").value),
    nb_chambres: Number(document.getElementById("chambres").value),
    nb_salles_bain: Number(document.getElementById("sdb").value),
    etage: Number(document.getElementById("etage").value),
  };

  try {
    const result = await fetchJSON("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const rangePosition = Math.min(
      100,
      Math.max(
        0,
        ((result.estimation - result.range_min) /
          Math.max(result.range_max - result.range_min, 1)) *
          100
      )
    );

    // ── Bloc résultat existant (inchangé) ────────────────────────────────────
    resultsContent.innerHTML = `
      <h2 style="color:var(--accent);">ESTIMATION MODELE ML</h2>
      <p style="margin-top:10px; color:var(--muted); font-size:0.9rem;">Resultat genere par le modele de Machine Learning.</p>
      
      <div class="result-range" style="margin-top:20px;">
        <span>${formatMoney(result.range_min)}</span>
        <div class="range-bar"><span class="range-dot" style="left:${rangePosition}%;"></span></div>
        <span>${formatMoney(result.range_max)}</span>
      </div>
      <p style="margin:20px 0; font-size:1.2rem;">Estimation Finale : <strong style="color:var(--text); font-size:1.6rem;">${formatMoney(result.estimation)}</strong></p>
      <hr style="border-color:var(--border);" />
      <p>Prix au m2 estime : <strong style="color:var(--accent);">${formatMoney(result.prix_m2_estime)} / m2</strong></p>
    `;

    // ── NOUVEAU : affiche le graphique de tendance si l'API renvoie un forecast ─
    if (result.forecast && result.forecast.length > 0) {
      renderTrendChart(result.forecast, result.taux_annuel, result.variation_5ans);
    }
    // ─────────────────────────────────────────────────────────────────────────

  } catch (error) {
    resultsContent.innerHTML = `<p style="color:#b91c1c;">Impossible de calculer l'estimation. Veuillez reessayer.</p>`;
  }
});

loadCities();