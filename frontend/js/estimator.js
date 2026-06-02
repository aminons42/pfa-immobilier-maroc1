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
    villeSelect.innerHTML = cities.map((city) => `<option value="${city.name}">${city.name}</option>`).join("");
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
    const quartiers = await fetchJSON(`/api/cities/${encodeURIComponent(city)}/quartiers`);
    quartierSelect.innerHTML = quartiers.map((q) => `<option value="${q.name}">${q.name}</option>`).join("");
    const preselect = getQueryParam("quartier");
    if (preselect) quartierSelect.value = preselect;
  } catch (error) {
    quartierSelect.innerHTML = "<option>Erreur de chargement</option>";
  } finally {
    quartierSelect.disabled = false;
  }
}

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

    const rangePosition = Math.min(100, Math.max(0, ((result.estimation - result.range_min) / Math.max(result.range_max - result.range_min, 1)) * 100));

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
  } catch (error) {
    resultsContent.innerHTML = `<p style="color:#b91c1c;">Impossible de calculer l'estimation. Veuillez reessayer.</p>`;
  }
});

loadCities();