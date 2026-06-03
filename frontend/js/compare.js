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
  const priceDiff  = percentDiff(a.avg_prix_m2, b.avg_prix_m2);
  const volumeDiff = percentDiff(a.listing_count, b.listing_count);
  const capitalDiff = percentDiff(a.avg_prix_total, b.avg_prix_total);

  const cheaper     = a.avg_prix_m2 <= b.avg_prix_m2 ? a : b;
  const pricier     = a.avg_prix_m2 >  b.avg_prix_m2 ? a : b;
  const busier      = a.listing_count >= b.listing_count ? a : b;
  const quieter     = a.listing_count <  b.listing_count ? a : b;
  const absDiff     = Math.abs(priceDiff);
  const absVol      = Math.abs(volumeDiff);
  const absCap      = Math.abs(capitalDiff);

  // 1. Prix d'achat
  let accessPhrase;
  if (absDiff < 5) {
    accessPhrase = `Le prix au m² est presque le même dans les deux villes. Le budget d'achat sera similaire peu importe votre choix.`;
  } else if (absDiff < 20) {
    accessPhrase = `<strong>${cheaper.city}</strong> coûte <strong>${absDiff}% moins cher</strong> au m². Avec le même budget, vous pouvez acheter une surface plus grande qu'à <strong>${pricier.city}</strong>.`;
  } else {
    accessPhrase = `Grand écart de prix : le m² à <strong>${cheaper.city}</strong> coûte <strong>${absDiff}% moins cher</strong> qu'à <strong>${pricier.city}</strong>. Si votre budget est limité, <strong>${cheaper.city}</strong> est nettement plus accessible.`;
  }

  // 2. Facilité de revente
  let reventePhrase;
  if (absVol < 10) {
    reventePhrase = `Le nombre d'annonces est proche dans les deux villes. Si vous avez besoin de revendre, les deux marchés offrent des conditions similaires.`;
  } else if (busier.listing_count > 2 * quieter.listing_count) {
    reventePhrase = `<strong>${busier.city}</strong> a beaucoup plus d'annonces actives que <strong>${quieter.city}</strong>. Cela veut dire plus d'acheteurs potentiels et une revente plus facile si vous en avez besoin.`;
  } else {
    reventePhrase = `<strong>${busier.city}</strong> a <strong>${absVol}% plus d'annonces</strong> en circulation. Plus de transactions signifie plus de facilité à revendre votre bien.`;
  }

  // 3. Loyer potentiel
  const rendementScore_a = (1 / a.avg_prix_m2) * a.avg_surface;
  const rendementScore_b = (1 / b.avg_prix_m2) * b.avg_surface;
  const bestRendement = rendementScore_a >= rendementScore_b ? a : b;
  const worstRendement = rendementScore_a >= rendementScore_b ? b : a;
  const loyerPhrase = `À <strong>${bestRendement.city}</strong>, le prix payé est moins élevé par rapport à la surface obtenue. En achetant pour louer, vos chances de couvrir vos charges mensuelles sont meilleures qu'à <strong>${worstRendement.city}</strong>.`;

  // 4. Budget total nécessaire
  let budgetPhrase;
  if (absCap < 5) {
    budgetPhrase = `Le montant moyen d'un bien est proche dans les deux villes. Pas de grande différence sur le budget total à prévoir.`;
  } else {
    const lessCapital = a.avg_prix_total <= b.avg_prix_total ? a : b;
    const moreCapital = a.avg_prix_total >  b.avg_prix_total ? a : b;
    budgetPhrase = `Un bien à <strong>${lessCapital.city}</strong> coûte en moyenne <strong>${absCap}% moins cher</strong> qu'à <strong>${moreCapital.city}</strong>. Avec le même apport, vous pouvez envisager d'acheter deux biens à <strong>${lessCapital.city}</strong> au lieu d'un seul.`;
  }

  // 5. Pour quel type d'investisseur
  function getProfile(city, other) {
    if (city.avg_prix_m2 > other.avg_prix_m2 && city.listing_count > other.listing_count) {
      return `convient à ceux qui veulent un bien facile à revendre et qui misent sur une hausse de valeur à long terme.`;
    } else if (city.avg_prix_m2 <= other.avg_prix_m2 && city.avg_surface >= other.avg_surface) {
      return `convient à ceux qui veulent acheter pour louer et générer un revenu mensuel régulier.`;
    } else if (city.listing_count < other.listing_count) {
      return `convient à ceux qui veulent conserver leur bien sur plusieurs années sans pression de revente.`;
    } else {
      return `convient à un profil varié, que ce soit pour habiter, louer ou revendre.`;
    }
  }

  const profilA = getProfile(a, b);
  const profilB = getProfile(b, a);

  verdictCard.innerHTML = `
    <div class="verdict-title">Verdict IA — Analyse Investisseur</div>
    <div class="verdict-body">

      <p>
        <em style="color:var(--muted); font-size:0.78rem; display:block; margin-bottom:3px;">PRIX D'ACHAT</em>
        ${accessPhrase}
      </p>

      <p style="margin-top:12px;">
        <em style="color:var(--muted); font-size:0.78rem; display:block; margin-bottom:3px;">FACILITE DE REVENTE</em>
        ${reventePhrase}
      </p>

      <p style="margin-top:12px;">
        <em style="color:var(--muted); font-size:0.78rem; display:block; margin-bottom:3px;">ACHAT POUR LOUER</em>
        ${loyerPhrase}
      </p>

      <p style="margin-top:12px;">
        <em style="color:var(--muted); font-size:0.78rem; display:block; margin-bottom:3px;">BUDGET TOTAL</em>
        ${budgetPhrase}
      </p>

      <div style="margin-top:16px; padding-top:14px; border-top:1px dashed var(--border);">
        <em style="color:var(--muted); font-size:0.78rem;">POUR QUEL INVESTISSEUR</em>
        <p style="margin-top:6px;">
          <strong>${a.city}</strong> — <span style="color:var(--muted);">${profilA}</span>
        </p>
        <p style="margin-top:4px;">
          <strong>${b.city}</strong> — <span style="color:var(--muted);">${profilB}</span>
        </p>
      </div>

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
