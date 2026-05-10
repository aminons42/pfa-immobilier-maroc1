import { fetchJSON, formatMoney } from "./utils.js";

const map = L.map("map").setView([31.5, -7.0], 6);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution: "",
}).addTo(map);

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
  if (max === min) return 10;
  return 6 + ((count - min) / (max - min)) * 14;
}

function buildStars(score) {
  const stars = Math.round(score);
  return "★".repeat(stars) + "☆".repeat(Math.max(0, 5 - stars));
}

async function loadHeatmap() {
  const data = await fetchJSON("/api/heatmap");
  const counts = data.cities.map((city) => city.listing_count);
  const minCount = Math.min(...counts);
  const maxCount = Math.max(...counts);

  data.cities.filter((city) => city.lat !== null && city.lng !== null).forEach((city, index) => {
    const delay = index * 100;
    setTimeout(() => {
      const marker = L.circleMarker([city.lat, city.lng], {
        radius: radiusScale(city.listing_count, minCount, maxCount),
        color: markerColor(city.avg_prix_m2),
        fillColor: markerColor(city.avg_prix_m2),
        fillOpacity: 0.7,
        className: "drop-marker",
      }).addTo(map);

      marker.bindTooltip(
        `<strong>${city.name}</strong><br/>${formatMoney(city.avg_prix_m2)} / m²<br/>Score: ${city.investment_score.toFixed(1)}`
      );

      marker.on("click", async () => {
        map.setView([city.lat, city.lng], 11, { animate: true });
        quartierLayer.clearLayers();
        const quartiers = await fetchJSON(`/api/cities/${city.name}/quartiers`);
        quartiers.forEach((q) => {
          const qMarker = L.circleMarker([city.lat + (Math.random() - 0.5) * 0.1, city.lng + (Math.random() - 0.5) * 0.1], {
            radius: 4,
            color: "#9aa4b2",
            fillOpacity: 0.6,
          }).addTo(quartierLayer);
          qMarker.on("click", () => {
            window.location.href = `estimator.html?ville=${encodeURIComponent(city.name)}&quartier=${encodeURIComponent(q.name)}`;
          });
        });

        panelTitle.textContent = city.name;
        panelMeta.innerHTML = `${buildStars(city.investment_score)} | ${formatMoney(city.avg_prix_m2)} / m² | ${city.listing_count} annonces | ${city.avg_surface.toFixed(0)} m²`;
        infoPanel.classList.add("active");

        btnDashboard.onclick = () => (window.location.href = `dashboard.html?ville=${encodeURIComponent(city.name)}`);
        btnEstimator.onclick = () => (window.location.href = `estimator.html?ville=${encodeURIComponent(city.name)}`);
        btnCompare.onclick = () => (window.location.href = `compare.html?ville=${encodeURIComponent(city.name)}`);
      });
    }, delay);
  });
}

loadHeatmap();
