# Canary Islands Earthquake Monitor

A visually stunning, real-time earthquake monitoring dashboard for the Canary Islands. This application displays live seismic data from the USGS Earthquake API with beautiful visualizations and interactive features.

## Features

- **Live Data**: Fetches real-time earthquake data from USGS API every 2 minutes
- **Interactive Map**: Dark-themed map focused on the Canary Islands with animated earthquake markers
- **Magnitude Visualization**: Color-coded markers and size scaling based on earthquake magnitude
- **Statistics Dashboard**: Total events, max magnitude, average depth, and recent activity
- **Time Filtering**: View earthquakes from the last 24 hours, 7 days, or 30 days
- **Earthquake List**: Scrollable list of recent earthquakes with quick details
- **Magnitude Distribution Chart**: Visual breakdown of earthquake magnitudes
- **Detail Modal**: Click any earthquake for comprehensive information
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **HTML5** - Semantic markup
- **CSS3** - Modern styling with CSS variables, animations, and gradients
- **Vanilla JavaScript** - No frameworks required
- **Leaflet.js** - Interactive mapping
- **IGN Spain GeoRSS Feed** - Real-time seismic data from Instituto Geográfico Nacional

## No Server Required

This is a completely client-side application. Simply open `index.html` in a web browser to run it. The app fetches data directly from the IGN Spain public GeoRSS feed.

## Quick Start

1. Clone or download this repository
2. Open `index.html` in your web browser
3. That's it! The app will automatically fetch and display earthquake data

Alternatively, you can serve it with any static file server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js (npx)
npx serve

# Using PHP
php -S localhost:8000
```

## Data Source

Earthquake data is provided by the [Instituto Geográfico Nacional (IGN) Spain](https://www.ign.es/web/ign/portal/sis-catalogo-terremotos). The IGN operates the Spanish National Seismic Network and provides real-time monitoring of seismic activity in Spain and the Canary Islands.

The GeoRSS feed provides the last 10 days of earthquakes, filtered for the Canary Islands region:

- Latitude: 27.0°N to 29.5°N
- Longitude: 19.0°W to 13.0°W

## Configuration

You can modify the configuration in `app.js`:

```javascript
const CONFIG = {
    bounds: {
        minLat: 27.0,
        maxLat: 29.5,
        minLon: -19.0,
        maxLon: -13.0
    },
    center: [28.3, -15.8],
    defaultZoom: 8,
    refreshInterval: 120000, // 2 minutes
    defaultDays: 1
};
```

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

MIT License - Feel free to use and modify as needed.
