import { fetchJSON, formatMoney } from "./utils.js";

const map = L.map("map").setView([31.5, -7.0], 6);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { attribution: "" }).addTo(map);

const infoPanel = document.getElementById("infoPanel");
const panelTitle = document.getElementById("panelTitle");
const panelMeta = document.getElementById("panelMeta");
const btnDashboard = document.getElementById("btnDashboard");
const btnEstimator = document.getElementById("btnEstimator");
const btnCompare = document.getElementById("btnCompare");

let quartierLayer = L.layerGroup().addTo(map);

function markerColor(avgPrix) {
  if (avgPrix < 10000) return "#2ECC71";
  if (avgPrix <= 13000) return "#C9A84C";
  return "#E74C3C";
}

function radiusScale(count, min, max) {
  if (max === min) return 12;
  return 7 + ((count - min) / (max - min)) * 15;
}

async function loadHeatmap() {
  try {
    const data = await fetchJSON("/api/heatmap");
    if (!data.cities || data.cities.length === 0) return;

    const counts = data.cities.map((city) => city.listing_count);
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);

    data.cities.forEach((city, index) => {
      const delay = index * 80;
      setTimeout(() => {
        const marker = L.circleMarker([city.lat, city.lng], {
          radius: radiusScale(city.listing_count, minCount, maxCount),
          color: markerColor(city.avg_prix_m2),
          fillColor: markerColor(city.avg_prix_m2),
          fillOpacity: 0.65,
          className: "drop-marker"
        }).addTo(map);

        marker.bindTooltip(`<strong>${city.name}</strong><br/>Moyenne : ${formatMoney(city.avg_prix_m2)} / m²`);

        marker.on("click", async () => {
          map.setView([city.lat, city.lng], 12, { animate: true });
          quartierLayer.clearLayers();
          
          try {
            const quartiers = await fetchJSON(`/api/cities/${encodeURIComponent(city.name)}/quartiers`);
            quartiers.forEach((q, idx) => {
              const angle = (idx / quartiers.length) * 2 * Math.PI;
              const r = 0.02 + Math.random() * 0.02;
              const lat = city.lat + r * Math.sin(angle);
              const lng = city.lng + r * Math.cos(angle);
              
              const qMarker = L.circleMarker([lat, lng], { radius: 5, color: "#C9A84C", fillColor: "#C9A84C", fillOpacity: 0.8, weight: 1 }).addTo(quartierLayer);
              
              qMarker.bindTooltip(`<strong>Quartier : ${q.name}</strong><br/><span style="color:var(--accent); font-size:0.8rem;">Cliquez pour ouvrir l'estimateur</span>`, { sticky: true });

              qMarker.on("click", () => {
                window.location.href = `estimator.html?ville=${encodeURIComponent(city.name)}&quartier=${encodeURIComponent(q.name)}`;
              });
            });
          } catch (error) {
            panelMeta.textContent = "Chargement des quartiers impossible.";
          }

          panelTitle.textContent = city.name;
          panelMeta.innerHTML = `Marche local | Prix moyen au m2 : ${formatMoney(city.avg_prix_m2)} | ${city.listing_count} annonces repertoriees`;
          infoPanel.classList.add("active");

          btnDashboard.onclick = () => (window.location.href = `dashboard.html?ville=${encodeURIComponent(city.name)}`);
          btnEstimator.onclick = () => (window.location.href = `estimator.html?ville=${encodeURIComponent(city.name)}`);
          btnCompare.onclick = () => (window.location.href = `compare.html?ville=${encodeURIComponent(city.name)}`);
        });
      }, delay);
    });
  } catch (error) {
    panelTitle.textContent = "Carte indisponible";
    panelMeta.textContent = "Impossible de charger les donnees.";
    infoPanel.classList.add("active");
  }
}

loadHeatmap();