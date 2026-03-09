// Canary Islands Earthquake Monitor
// Client-side only - No server required
// Data source: Instituto Geográfico Nacional (IGN) Spain

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        // Canary Islands bounding box (with padding for offshore events)
        bounds: {
            minLat: 27.0,
            maxLat: 29.5,
            minLon: -19.0,
            maxLon: -13.0
        },
        // Map center (approximately center of Canary Islands)
        center: [28.3, -15.8],
        defaultZoom: 8,
        // IGN Spain Canary Islands earthquake page (last 10 days, all magnitudes)
        ignCanariasUrl: 'https://www.ign.es/web/vlc-ultimo-terremoto/-/terremotos-canarias/get10dias',
        // Refresh interval (2 minutes)
        refreshInterval: 120000,
        // Default time range in days
        defaultDays: 1
    };

    // State
    let map = null;
    let earthquakeMarkers = [];
    let earthquakeData = [];
    let currentDays = CONFIG.defaultDays;
    let refreshTimer = null;

    // DOM Elements
    const elements = {
        map: document.getElementById('map'),
        lastUpdate: document.getElementById('lastUpdate'),
        totalQuakes: document.getElementById('totalQuakes'),
        maxMagnitude: document.getElementById('maxMagnitude'),
        avgDepth: document.getElementById('avgDepth'),
        recentCount: document.getElementById('recentCount'),
        earthquakeList: document.getElementById('earthquakeList'),
        detailModal: document.getElementById('detailModal'),
        closeModal: document.getElementById('closeModal'),
        modalMagnitude: document.getElementById('modalMagnitude'),
        modalTitle: document.getElementById('modalTitle'),
        modalTime: document.getElementById('modalTime'),
        modalDepth: document.getElementById('modalDepth'),
        modalLocation: document.getElementById('modalLocation'),
        modalCoords: document.getElementById('modalCoords'),
        modalStatus: document.getElementById('modalStatus'),
        modalLink: document.getElementById('modalLink'),
        filterBtns: document.querySelectorAll('.filter-btn')
    };

    // Initialize the application
    function init() {
        initMap();
        initEventListeners();
        fetchEarthquakeData();
        startAutoRefresh();
    }

    // Initialize Leaflet map
    function initMap() {
        map = L.map('map', {
            center: CONFIG.center,
            zoom: CONFIG.defaultZoom,
            minZoom: 7,
            maxZoom: 12,
            zoomControl: false
        });

        // Add zoom control to top-right corner
        L.control.zoom({
            position: 'topright'
        }).addTo(map);

        // Dark theme map tiles (CartoDB Dark Matter)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);

        // Set max bounds to keep focus on Canary Islands
        const bounds = L.latLngBounds(
            [CONFIG.bounds.minLat - 1, CONFIG.bounds.minLon - 1],
            [CONFIG.bounds.maxLat + 1, CONFIG.bounds.maxLon + 1]
        );
        map.setMaxBounds(bounds);
    }

    // Initialize event listeners
    function initEventListeners() {
        // Time filter buttons
        elements.filterBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                elements.filterBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentDays = parseInt(btn.dataset.days);
                fetchEarthquakeData();
            });
        });

        // Modal close
        elements.closeModal.addEventListener('click', closeModal);
        elements.detailModal.addEventListener('click', (e) => {
            if (e.target === elements.detailModal) {
                closeModal();
            }
        });

        // Keyboard escape to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
    }

    // Fetch earthquake data from IGN Spain Canary Islands page
    async function fetchEarthquakeData() {
        showLoading();

        try {
            const response = await fetch(CONFIG.ignCanariasUrl);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const htmlText = await response.text();
            const allQuakes = parseIGNCanariasHTML(htmlText);

            // Filter by time range
            const cutoffTime = Date.now() - currentDays * 24 * 60 * 60 * 1000;
            earthquakeData = allQuakes.filter(quake => quake.properties.time >= cutoffTime);
            
            // Sort by time (newest first)
            earthquakeData.sort((a, b) => b.properties.time - a.properties.time);
            
            updateUI();
            updateLastUpdateTime();
        } catch (error) {
            console.error('Error fetching earthquake data:', error);
            showError();
        }
    }

    // Parse IGN Canary Islands HTML page to extract earthquake data
    function parseIGNCanariasHTML(htmlText) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(htmlText, 'text/html');
        const rows = doc.querySelectorAll('tr');
        const earthquakes = [];

        rows.forEach(row => {
            try {
                const cells = row.querySelectorAll('td');
                if (cells.length < 10) return; // Skip header or invalid rows

                const eventId = cells[0]?.textContent?.trim() || '';
                const dateStr = cells[1]?.textContent?.trim() || '';
                const timeStr = cells[2]?.textContent?.trim() || '';
                const lat = parseFloat(cells[3]?.textContent?.trim() || '0');
                const lon = parseFloat(cells[4]?.textContent?.trim() || '0');
                const depth = parseFloat(cells[5]?.textContent?.trim() || '0');
                const magnitude = parseFloat(cells[7]?.textContent?.trim() || '0');
                const location = cells[9]?.textContent?.trim() || 'Canary Islands';

                // Skip if no valid coordinates
                if (lat === 0 && lon === 0) return;
                if (!eventId.startsWith('es')) return; // Skip non-event rows

                // Parse date: DD/MM/YYYY and time: HH:MM:SS
                const dateMatch = dateStr.match(/(\d{2})\/(\d{2})\/(\d{4})/);
                const timeMatch = timeStr.match(/(\d{2}):(\d{2}):(\d{2})/);
                
                let timestamp = Date.now();
                if (dateMatch && timeMatch) {
                    const [, day, month, year] = dateMatch;
                    const [, hour, minute, second] = timeMatch;
                    timestamp = new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}Z`).getTime();
                }

                const detailUrl = `https://www.ign.es/web/ign/portal/sis-catalogo-terremotos/-/catalogo-terremotos/detailTerremoto?evid=${eventId}`;

                earthquakes.push({
                    type: 'Feature',
                    id: eventId,
                    geometry: {
                        type: 'Point',
                        coordinates: [lon, lat, depth]
                    },
                    properties: {
                        mag: magnitude,
                        place: formatIGNLocation(location),
                        time: timestamp,
                        url: detailUrl,
                        status: 'reviewed',
                        depth: depth,
                        title: `M ${magnitude.toFixed(1)} - ${formatIGNLocation(location)}`
                    }
                });
            } catch (e) {
                console.warn('Error parsing earthquake row:', e);
            }
        });

        return earthquakes;
    }

    // Format IGN location codes to readable names
    function formatIGNLocation(code) {
        // Common Canary Islands location mappings
        const locationMap = {
            'ATLÁNTICO-CANARIAS': 'Atlantic Ocean - Canary Islands',
            'ATLANTICO-CANARIAS': 'Atlantic Ocean - Canary Islands',
            'TENERIFE': 'Tenerife',
            'GRAN CANARIA': 'Gran Canaria',
            'LANZAROTE': 'Lanzarote',
            'FUERTEVENTURA': 'Fuerteventura',
            'LA PALMA': 'La Palma',
            'LA GOMERA': 'La Gomera',
            'EL HIERRO': 'El Hierro'
        };

        const upperCode = code.toUpperCase();
        
        // Check for exact match
        if (locationMap[upperCode]) {
            return locationMap[upperCode];
        }

        // Check for partial matches
        for (const [key, value] of Object.entries(locationMap)) {
            if (upperCode.includes(key) || key.includes(upperCode)) {
                return value;
            }
        }

        // Clean up the code for display
        return code
            .replace(/\.[A-Z]+$/, '') // Remove suffix like .TF
            .replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    // Update all UI components
    function updateUI() {
        clearMarkers();
        updateMarkers();
        updateStats();
        updateEarthquakeList();
        updateMagnitudeChart();
    }

    // Clear existing markers
    function clearMarkers() {
        earthquakeMarkers.forEach(marker => map.removeLayer(marker));
        earthquakeMarkers = [];
    }

    // Update map markers
    function updateMarkers() {
        earthquakeData.forEach((quake, index) => {
            const coords = quake.geometry.coordinates;
            const props = quake.properties;
            const magnitude = props.mag || 0;
            const depth = coords[2] || 0;

            // Create custom marker
            const marker = createEarthquakeMarker(
                [coords[1], coords[0]],
                magnitude,
                depth,
                quake
            );

            marker.addTo(map);
            earthquakeMarkers.push(marker);
        });

        // Force map to invalidate size and redraw after markers are added
        setTimeout(() => {
            map.invalidateSize();
        }, 100);
    }

    // Create custom earthquake marker
    function createEarthquakeMarker(latlng, magnitude, depth, quakeData) {
        const size = getMarkerSize(magnitude);
        const color = getMagnitudeColor(magnitude);

        const markerHtml = `
            <div class="earthquake-marker">
                <div class="marker-inner" style="
                    width: ${size * 2}px;
                    height: ${size * 2}px;
                    background: ${color};
                    opacity: 0.3;
                "></div>
                <div class="marker-core" style="
                    width: ${size}px;
                    height: ${size}px;
                    background: ${color};
                    box-shadow: 0 0 ${size}px ${color};
                "></div>
            </div>
        `;

        const icon = L.divIcon({
            html: markerHtml,
            className: 'custom-earthquake-icon',
            iconSize: [size * 2, size * 2],
            iconAnchor: [size, size]
        });

        const marker = L.marker(latlng, { icon });

        // Popup content
        const popupContent = `
            <div style="
                font-family: 'Inter', sans-serif;
                padding: 8px;
                min-width: 180px;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 10px;
                ">
                    <div style="
                        width: 40px;
                        height: 40px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: ${color};
                        border-radius: 6px;
                        color: white;
                        font-weight: 700;
                        font-size: 14px;
                    ">${magnitude.toFixed(1)}</div>
                    <div style="flex: 1;">
                        <div style="font-weight: 600; font-size: 13px; color: #1a1a25;">
                            M ${magnitude.toFixed(1)} Earthquake
                        </div>
                        <div style="font-size: 11px; color: #606070;">
                            Depth: ${depth.toFixed(1)} km
                        </div>
                    </div>
                </div>
                <div style="font-size: 12px; color: #606070; margin-bottom: 8px;">
                    ${formatLocation(quakeData.properties.place)}
                </div>
                <div style="font-size: 11px; color: #a0a0b0;">
                    ${formatTime(quakeData.properties.time)}
                </div>
            </div>
        `;

        marker.bindPopup(popupContent, {
            className: 'custom-popup'
        });

        // Click handler for detail modal
        marker.on('click', () => {
            showQuakeDetail(quakeData);
        });

        return marker;
    }

    // Get marker size based on magnitude
    function getMarkerSize(magnitude) {
        if (magnitude < 2) return 12;
        if (magnitude < 3) return 16;
        if (magnitude < 4) return 22;
        if (magnitude < 5) return 30;
        return 40;
    }

    // Get color based on magnitude
    function getMagnitudeColor(magnitude) {
        if (magnitude < 2) return '#22c55e';
        if (magnitude < 3) return '#84cc16';
        if (magnitude < 4) return '#eab308';
        if (magnitude < 5) return '#f97316';
        return '#ef4444';
    }

    // Get magnitude class
    function getMagnitudeClass(magnitude) {
        if (magnitude < 2) return 'mag-1';
        if (magnitude < 3) return 'mag-2';
        if (magnitude < 4) return 'mag-3';
        if (magnitude < 5) return 'mag-4';
        return 'mag-5';
    }

    // Update statistics
    function updateStats() {
        const total = earthquakeData.length;
        const magnitudes = earthquakeData.map(q => q.properties.mag || 0);
        const depths = earthquakeData.map(q => q.properties.depth || q.geometry.coordinates[2] || 0);
        
        const maxMag = magnitudes.length > 0 ? Math.max(...magnitudes) : 0;
        const avgDepth = depths.length > 0 
            ? depths.reduce((a, b) => a + b, 0) / depths.length 
            : 0;

        // Count earthquakes in last 6 hours
        const sixHoursAgo = Date.now() - 6 * 60 * 60 * 1000;
        const recentQuakes = earthquakeData.filter(q => q.properties.time > sixHoursAgo);

        // Animate stat updates
        animateValue(elements.totalQuakes, total);
        elements.maxMagnitude.textContent = maxMag.toFixed(1);
        animateValue(elements.avgDepth, Math.round(avgDepth));
        animateValue(elements.recentCount, recentQuakes.length);
    }

    // Animate number value
    function animateValue(element, newValue) {
        const currentValue = parseInt(element.textContent) || 0;
        const diff = newValue - currentValue;
        const steps = 20;
        const stepValue = diff / steps;
        let step = 0;

        const animate = () => {
            step++;
            const value = Math.round(currentValue + stepValue * step);
            element.textContent = value;
            
            if (step < steps) {
                requestAnimationFrame(animate);
            } else {
                element.textContent = newValue;
            }
        };

        if (diff !== 0) {
            requestAnimationFrame(animate);
        }
    }

    // Update earthquake list
    function updateEarthquakeList() {
        if (earthquakeData.length === 0) {
            elements.earthquakeList.innerHTML = `
                <div class="empty-state">
                    <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <path d="M12 6v6l4 2"/>
                    </svg>
                    <span class="empty-text">No earthquakes detected in this time period</span>
                </div>
            `;
            return;
        }

        const listHtml = earthquakeData.slice(0, 50).map((quake, index) => {
            const props = quake.properties;
            const coords = quake.geometry.coordinates;
            const magnitude = props.mag || 0;
            const depth = props.depth || coords[2] || 0;
            const magClass = getMagnitudeClass(magnitude);

            return `
                <div class="earthquake-item" data-index="${index}">
                    <div class="quake-magnitude ${magClass}">${magnitude.toFixed(1)}</div>
                    <div class="quake-info">
                        <div class="quake-location">${formatLocation(props.place)}</div>
                        <div class="quake-meta">
                            <span class="quake-time">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="12" cy="12" r="10"/>
                                    <path d="M12 6v6l4 2"/>
                                </svg>
                                ${formatTimeAgo(props.time)}
                            </span>
                            <span class="quake-depth">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <path d="M12 2v20M2 12h20"/>
                                </svg>
                                ${depth.toFixed(1)} km
                            </span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        elements.earthquakeList.innerHTML = listHtml;

        // Add click handlers
        elements.earthquakeList.querySelectorAll('.earthquake-item').forEach(item => {
            item.addEventListener('click', () => {
                const index = parseInt(item.dataset.index);
                const quake = earthquakeData[index];
                showQuakeDetail(quake);
                
                // Pan to earthquake location
                const coords = quake.geometry.coordinates;
                map.panTo([coords[1], coords[0]]);
            });
        });
    }

    // Update magnitude distribution chart
    function updateMagnitudeChart() {
        const distribution = {
            'mag-1': 0,
            'mag-2': 0,
            'mag-3': 0,
            'mag-4': 0,
            'mag-5': 0
        };

        earthquakeData.forEach(quake => {
            const mag = quake.properties.mag || 0;
            const magClass = getMagnitudeClass(mag);
            distribution[magClass]++;
        });

        const maxCount = Math.max(...Object.values(distribution), 1);
        const labels = ['<2.0', '2-3', '3-4', '4-5', '>5.0'];

        const chartHtml = Object.entries(distribution).map(([magClass, count], index) => {
            const height = (count / maxCount) * 60;
            return `
                <div class="chart-bar-wrapper">
                    <div class="chart-bar ${magClass}" 
                         style="height: ${height}px" 
                         data-count="${count}"></div>
                    <span class="chart-label">${labels[index]}</span>
                </div>
            `;
        }).join('');

        document.querySelector('.chart-bars').innerHTML = chartHtml;
    }

    // Show earthquake detail modal
    function showQuakeDetail(quake) {
        const props = quake.properties;
        const coords = quake.geometry.coordinates;
        const magnitude = props.mag || 0;
        const depth = coords[2] || 0;
        const color = getMagnitudeColor(magnitude);

        elements.modalMagnitude.textContent = magnitude.toFixed(1);
        elements.modalMagnitude.style.background = color;
        elements.modalTitle.textContent = formatLocation(props.place) || 'Unknown Location';
        elements.modalTime.textContent = formatTime(props.time);
        elements.modalDepth.textContent = `${depth.toFixed(1)} km`;
        elements.modalLocation.textContent = props.place || 'Unknown';
        elements.modalCoords.textContent = `${coords[1].toFixed(4)}°N, ${Math.abs(coords[0]).toFixed(4)}°W`;
        elements.modalStatus.textContent = props.status || 'Unknown';
        elements.modalLink.href = props.url || '#';

        elements.detailModal.classList.remove('hidden');
    }

    // Close modal
    function closeModal() {
        elements.detailModal.classList.add('hidden');
    }

    // Format location string
    function formatLocation(place) {
        if (!place) return 'Canary Islands Region';
        // Clean up USGS location format
        return place.replace(/^\d+\s*km\s+\w+\s+of\s+/i, '').trim() || place;
    }

    // Format timestamp
    function formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString('en-GB', {
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    }

    // Format time ago
    function formatTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        
        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
        return formatTime(timestamp);
    }

    // Update last update time
    function updateLastUpdateTime() {
        const now = new Date();
        elements.lastUpdate.textContent = now.toLocaleTimeString('en-GB', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    }

    // Show loading state
    function showLoading() {
        elements.earthquakeList.innerHTML = `
            <div class="loading-state">
                <div class="spinner"></div>
                <span>Loading earthquake data...</span>
            </div>
        `;
    }

    // Show error state
    function showError() {
        elements.earthquakeList.innerHTML = `
            <div class="empty-state">
                <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 8v4M12 16h.01"/>
                </svg>
                <span class="empty-text">Failed to load earthquake data. Will retry automatically.</span>
            </div>
        `;
    }

    // Start auto-refresh
    function startAutoRefresh() {
        if (refreshTimer) {
            clearInterval(refreshTimer);
        }
        refreshTimer = setInterval(fetchEarthquakeData, CONFIG.refreshInterval);
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
