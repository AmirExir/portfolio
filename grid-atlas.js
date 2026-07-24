(() => {
  'use strict';

  const MANIFEST_URL = 'ERCOTAPI/grid_atlas_data/manifest.json?v=20260723-voltage-defaults';
  const SOURCE_IDS = ['atlas-boundaries', 'atlas-lines', 'atlas-substations', 'atlas-plants', 'atlas-highlight'];
  const LAYER_IDS = [
    'atlas-boundary-fill',
    'atlas-boundary-line',
    'atlas-lines-layer',
    'atlas-substations-layer',
    'atlas-plants-layer',
    'atlas-highlight-line',
    'atlas-highlight-point',
  ];
  const NUMBER_FORMAT = new Intl.NumberFormat('en-US');
  const query = new URLSearchParams(window.location.search);
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (query.get('embed') === '1') {
    document.documentElement.classList.add('is-embed');
  }

  const elements = {
    regionButtons: document.getElementById('regionButtons'),
    minimumVoltage: document.getElementById('minimumVoltage'),
    minimumPlantMw: document.getElementById('minimumPlantMw'),
    showLines: document.getElementById('showLines'),
    showSubstations: document.getElementById('showSubstations'),
    showPlants: document.getElementById('showPlants'),
    showBoundaries: document.getElementById('showBoundaries'),
    includeUnknownVoltage: document.getElementById('includeUnknownVoltage'),
    assetSearchForm: document.getElementById('assetSearchForm'),
    assetSearch: document.getElementById('assetSearch'),
    status: document.getElementById('atlasStatus'),
    metrics: document.getElementById('atlasMetrics'),
    loadingPanel: document.getElementById('loadingPanel'),
    loadingTitle: document.getElementById('loadingTitle'),
    loadingDetail: document.getElementById('loadingDetail'),
    loadingFallback: document.getElementById('loadingFallback'),
    disclaimer: document.getElementById('atlasDisclaimer'),
    regionSource: document.getElementById('regionSource'),
  };

  let manifest = null;
  let regionsById = new Map();
  let currentRegion = null;
  let currentCollections = null;
  let searchIndex = [];
  let activeRequest = null;
  let regionRequestSequence = 0;
  let popup = null;

  const emptyCollection = () => ({ type: 'FeatureCollection', features: [] });

  function atlasStyle() {
    return {
      version: 8,
      sources: {
        'carto-dark': {
          type: 'raster',
          tiles: ['https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'],
          tileSize: 256,
          attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
          maxzoom: 20,
        },
      },
      layers: [
        {
          id: 'atlas-background',
          type: 'background',
          paint: { 'background-color': '#07111f' },
        },
        {
          id: 'carto-dark-layer',
          type: 'raster',
          source: 'carto-dark',
          minzoom: 0,
          maxzoom: 20,
        },
      ],
    };
  }

  function showFatalError(title, detail) {
    elements.loadingPanel.hidden = false;
    elements.loadingPanel.classList.add('is-error');
    elements.loadingTitle.textContent = title;
    elements.loadingDetail.textContent = detail;
    elements.loadingFallback.hidden = false;
    elements.status.textContent = detail;
  }

  if (!window.maplibregl) {
    showFatalError(
      'The interactive map library did not load.',
      'Check your connection or open the ERCOT dashboard instead.',
    );
    return;
  }

  const map = new window.maplibregl.Map({
    container: 'gridMap',
    style: atlasStyle(),
    center: [-99.02, 30.95],
    zoom: 4.45,
    minZoom: 1,
    maxZoom: 17,
    cooperativeGestures: true,
    attributionControl: false,
    fadeDuration: prefersReducedMotion ? 0 : 300,
  });

  map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), 'top-right');
  map.addControl(
    new window.maplibregl.AttributionControl({ compact: true }),
    'bottom-right',
  );

  const mapReady = new Promise((resolve, reject) => {
    map.once('load', resolve);
    map.once('error', (event) => {
      if (!map.loaded()) {
        reject(event.error || new Error('Map renderer failed to load'));
      }
    });
  });

  function setLoading(title, detail) {
    elements.loadingPanel.hidden = false;
    elements.loadingPanel.classList.remove('is-error');
    elements.loadingTitle.textContent = title;
    elements.loadingDetail.textContent = detail;
    elements.loadingFallback.hidden = true;
    elements.status.textContent = detail;
  }

  function hideLoading() {
    elements.loadingPanel.hidden = true;
    elements.loadingPanel.classList.remove('is-error');
    elements.loadingFallback.hidden = true;
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
  }

  function firstUseful(values, fallback = '') {
    for (const value of values) {
      if (value !== null && value !== undefined && String(value).trim()) {
        return String(value).trim();
      }
    }
    return fallback;
  }

  function voltageBand(voltage) {
    if (voltage === null) return 'Unknown voltage';
    if (voltage >= 500) return '500–765 kV';
    if (voltage >= 345) return '345–499 kV';
    if (voltage >= 230) return '230–344 kV';
    if (voltage >= 138) return '138–229 kV';
    if (voltage >= 69) return '69–137 kV';
    return 'Below 69 kV';
  }

  function normalizedPaths(paths) {
    if (!Array.isArray(paths)) return [];
    return paths
      .filter((path) => Array.isArray(path) && path.length >= 2)
      .map((path) => path.filter(
        (point) => Array.isArray(point)
          && Number.isFinite(Number(point[0]))
          && Number.isFinite(Number(point[1])),
      ))
      .filter((path) => path.length >= 2);
  }

  function pointGeometry(record) {
    const lon = Number(record.lon);
    const lat = Number(record.lat);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
    return { type: 'Point', coordinates: [lon, lat] };
  }

  function sourceLabel(record) {
    return firstUseful([record.source, record.utility, record.owner], 'Public GIS source');
  }

  function lineFeature(record, index) {
    const paths = normalizedPaths(record.paths);
    if (!paths.length) return null;
    const voltage = finiteNumber(record.voltage);
    const properties = {
      feature_id: firstUseful([record.asset_id, record.id], `line-${index}`),
      asset_type: 'Transmission line',
      name: firstUseful([record.name], 'Unnamed transmission line'),
      band: voltageBand(voltage),
      owner: firstUseful([record.owner]),
      operator: firstUseful([record.operator]),
      status: firstUseful([record.status]),
      source: sourceLabel(record),
      source_url: firstUseful([record.source_url]),
      voltage_source: firstUseful([record.voltage_source]),
      voltage_source_url: firstUseful([record.voltage_source_url]),
      voltage_match_status: firstUseful([record.voltage_match_status]),
      voltage_match_confidence: finiteNumber(record.voltage_match_confidence),
      voltage_retrieved_at: firstUseful([record.voltage_retrieved_at]),
      country: firstUseful([record.country]),
    };
    if (voltage !== null) properties.voltage = voltage;
    const feature = {
      type: 'Feature',
      properties,
      geometry: { type: 'MultiLineString', coordinates: paths },
    };
    const firstPoint = paths[0][Math.floor(paths[0].length / 2)];
    return {
      feature,
      search: {
        kind: 'line',
        label: properties.name,
        text: [
          properties.name,
          properties.owner,
          properties.operator,
          properties.status,
          properties.country,
          voltage === null ? 'unknown voltage' : `${voltage} kv`,
        ].join(' ').toLowerCase(),
        coordinates: firstPoint,
        properties,
        geometry: feature.geometry,
      },
    };
  }

  function substationFeature(record, index) {
    const geometry = pointGeometry(record);
    if (!geometry) return null;
    const voltage = finiteNumber(record.max_voltage ?? record.voltage);
    const properties = {
      feature_id: firstUseful([record.asset_id, record.id], `substation-${index}`),
      asset_type: 'Substation',
      name: firstUseful([record.name], 'Unnamed substation'),
      category: firstUseful([record.type], 'Substation'),
      status: firstUseful([record.status]),
      owner: firstUseful([record.owner]),
      operator: firstUseful([record.operator]),
      place: [record.city, record.county, record.state].filter(Boolean).join(', '),
      source: sourceLabel(record),
      source_url: firstUseful([record.source_url]),
      voltage_source: firstUseful([record.voltage_source]),
      voltage_source_url: firstUseful([record.voltage_source_url]),
      voltage_match_status: firstUseful([record.voltage_match_status]),
      voltage_match_confidence: finiteNumber(record.voltage_match_confidence),
      voltage_retrieved_at: firstUseful([record.voltage_retrieved_at]),
      country: firstUseful([record.country]),
    };
    if (voltage !== null) properties.voltage = voltage;
    const feature = { type: 'Feature', properties, geometry };
    return {
      feature,
      search: {
        kind: 'point',
        label: properties.name,
        text: [
          properties.name,
          properties.category,
          properties.status,
          properties.owner,
          properties.operator,
          properties.place,
          properties.country,
          voltage === null ? 'unknown voltage' : `${voltage} kv`,
        ].join(' ').toLowerCase(),
        coordinates: geometry.coordinates,
        properties,
        geometry,
      },
    };
  }

  function plantFeature(record, index) {
    const geometry = pointGeometry(record);
    if (!geometry) return null;
    const capacity = finiteNumber(record.capacity_mw ?? record.summer_mw ?? record.installed_mw);
    const properties = {
      feature_id: firstUseful([record.asset_id, record.plant_code, record.id], `plant-${index}`),
      asset_type: 'Power plant',
      name: firstUseful([record.name], 'Unnamed power plant'),
      category: firstUseful([record.fuel, record.technology], 'Other / unknown'),
      status: firstUseful([record.status]),
      place: [record.city, record.county, record.state].filter(Boolean).join(', '),
      source: sourceLabel(record),
      source_url: firstUseful([record.source_url]),
      country: firstUseful([record.country]),
    };
    if (capacity !== null) properties.capacity_mw = capacity;
    const feature = { type: 'Feature', properties, geometry };
    return {
      feature,
      search: {
        kind: 'point',
        label: properties.name,
        text: [
          properties.name,
          properties.category,
          properties.place,
          properties.country,
          capacity === null ? '' : `${capacity} mw`,
        ].join(' ').toLowerCase(),
        coordinates: geometry.coordinates,
        properties,
        geometry,
      },
    };
  }

  function boundaryFeature(record, index) {
    const polygons = Array.isArray(record.polygons) ? record.polygons : [];
    const coordinates = polygons
      .map((polygon) => {
        const outer = Array.isArray(polygon.outer) ? polygon.outer : [];
        const holes = Array.isArray(polygon.holes) ? polygon.holes : [];
        if (outer.length < 3) return null;
        return [outer, ...holes.filter((ring) => Array.isArray(ring) && ring.length >= 3)];
      })
      .filter(Boolean);
    if (!coordinates.length) return null;
    return {
      type: 'Feature',
      properties: {
        feature_id: firstUseful([record.id], `boundary-${index}`),
        label: firstUseful([record.label], 'Regional footprint'),
        kind: firstUseful([record.kind], 'region'),
        source_url: firstUseful([record.source_url]),
      },
      geometry: { type: 'MultiPolygon', coordinates },
    };
  }

  function buildCollections(payload) {
    const lines = [];
    const substations = [];
    const plants = [];
    const boundaries = [];
    const searchable = [];

    (payload.transmission_lines || []).forEach((record, index) => {
      const converted = lineFeature(record, index);
      if (!converted) return;
      lines.push(converted.feature);
      searchable.push(converted.search);
    });
    (payload.substations || []).forEach((record, index) => {
      const converted = substationFeature(record, index);
      if (!converted) return;
      substations.push(converted.feature);
      searchable.push(converted.search);
    });
    (payload.power_plants || []).forEach((record, index) => {
      const converted = plantFeature(record, index);
      if (!converted) return;
      plants.push(converted.feature);
      searchable.push(converted.search);
    });
    (payload.boundaries || []).forEach((record, index) => {
      const feature = boundaryFeature(record, index);
      if (feature) boundaries.push(feature);
    });

    searchIndex = searchable;
    return {
      lines: { type: 'FeatureCollection', features: lines },
      substations: { type: 'FeatureCollection', features: substations },
      plants: { type: 'FeatureCollection', features: plants },
      boundaries: { type: 'FeatureCollection', features: boundaries },
    };
  }

  function removeAtlasData() {
    if (!map.loaded()) return;
    LAYER_IDS.slice().reverse().forEach((id) => {
      if (map.getLayer(id)) map.removeLayer(id);
    });
    SOURCE_IDS.slice().reverse().forEach((id) => {
      if (map.getSource(id)) map.removeSource(id);
    });
  }

  function addAtlasData(collections, region) {
    removeAtlasData();
    map.addSource('atlas-boundaries', { type: 'geojson', data: collections.boundaries });
    map.addSource('atlas-lines', { type: 'geojson', data: collections.lines });
    map.addSource('atlas-substations', { type: 'geojson', data: collections.substations });
    map.addSource('atlas-plants', { type: 'geojson', data: collections.plants });
    map.addSource('atlas-highlight', { type: 'geojson', data: emptyCollection() });

    map.addLayer({
      id: 'atlas-boundary-fill',
      type: 'fill',
      source: 'atlas-boundaries',
      paint: {
        'fill-color': ['match', ['get', 'kind'], 'nerc', '#a78bfa', '#f5b942'],
        'fill-opacity': 0.045,
      },
    });
    map.addLayer({
      id: 'atlas-boundary-line',
      type: 'line',
      source: 'atlas-boundaries',
      paint: {
        'line-color': ['match', ['get', 'kind'], 'nerc', '#c4b5fd', '#f5b942'],
        'line-width': 1.7,
        'line-opacity': 0.9,
      },
    });
    map.addLayer({
      id: 'atlas-lines-layer',
      type: 'line',
      source: 'atlas-lines',
      paint: {
        'line-color': [
          'match',
          ['get', 'band'],
          '500–765 kV', '#f97316',
          '345–499 kV', '#22c55e',
          '230–344 kV', '#a855f7',
          '138–229 kV', '#0ea5e9',
          '69–137 kV', '#ef4444',
          'Below 69 kV', '#64748b',
          '#94a3b8',
        ],
        'line-width': [
          'match',
          ['get', 'band'],
          '500–765 kV', 2.4,
          '345–499 kV', 2.1,
          '230–344 kV', 1.7,
          1.15,
        ],
        'line-opacity': 0.78,
      },
    });
    map.addLayer({
      id: 'atlas-substations-layer',
      type: 'circle',
      source: 'atlas-substations',
      minzoom: region.id === 'all' ? 3 : 1.8,
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['coalesce', ['get', 'voltage'], 69],
          69, 2.2,
          230, 3.2,
          765, 5.2,
        ],
        'circle-color': '#22d3ee',
        'circle-opacity': 0.82,
        'circle-stroke-color': '#e6fffb',
        'circle-stroke-width': 0.45,
      },
    });
    map.addLayer({
      id: 'atlas-plants-layer',
      type: 'circle',
      source: 'atlas-plants',
      paint: {
        'circle-radius': [
          'interpolate',
          ['linear'],
          ['coalesce', ['get', 'capacity_mw'], 0],
          0, 3.2,
          500, 5.4,
          2000, 8.5,
        ],
        'circle-color': '#34d399',
        'circle-opacity': 0.86,
        'circle-stroke-color': '#eafff7',
        'circle-stroke-width': 0.6,
      },
    });
    map.addLayer({
      id: 'atlas-highlight-line',
      type: 'line',
      source: 'atlas-highlight',
      filter: ['==', ['geometry-type'], 'LineString'],
      paint: {
        'line-color': '#ffffff',
        'line-width': 5,
        'line-opacity': 0.96,
      },
    });
    map.addLayer({
      id: 'atlas-highlight-point',
      type: 'circle',
      source: 'atlas-highlight',
      filter: ['==', ['geometry-type'], 'Point'],
      paint: {
        'circle-radius': 10,
        'circle-color': '#ffffff',
        'circle-opacity': 0.3,
        'circle-stroke-color': '#ffffff',
        'circle-stroke-width': 2.5,
      },
    });
    applyFilters();
  }

  function voltageFilter() {
    const minimum = Number(elements.minimumVoltage.value) || 0;
    const known = [
      'all',
      ['has', 'voltage'],
      ['>=', ['to-number', ['get', 'voltage']], minimum],
    ];
    if (elements.includeUnknownVoltage.checked) {
      return ['any', known, ['!', ['has', 'voltage']]];
    }
    return known;
  }

  function setLayerVisibility(id, visible) {
    if (map.getLayer(id)) {
      map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
    }
  }

  function matchingCounts() {
    if (!currentCollections) return { lines: 0, substations: 0, plants: 0 };
    const minimumVoltage = Number(elements.minimumVoltage.value) || 0;
    const includeUnknown = elements.includeUnknownVoltage.checked;
    const minimumPlantMw = Number(elements.minimumPlantMw.value) || 0;
    const allowedVoltage = (feature) => {
      const value = finiteNumber(feature.properties.voltage);
      return value === null ? includeUnknown : value >= minimumVoltage;
    };
    return {
      lines: currentCollections.lines.features.filter(allowedVoltage).length,
      substations: currentCollections.substations.features.filter(allowedVoltage).length,
      plants: currentCollections.plants.features.filter((feature) => {
        const capacity = finiteNumber(feature.properties.capacity_mw);
        return (capacity ?? 0) >= minimumPlantMw;
      }).length,
    };
  }

  function renderMetrics() {
    const counts = matchingCounts();
    const values = [
      ['Lines', elements.showLines.checked ? counts.lines : 0],
      ['Substations', elements.showSubstations.checked ? counts.substations : 0],
      ['Plants', elements.showPlants.checked ? counts.plants : 0],
    ];
    elements.metrics.replaceChildren();
    values.forEach(([label, value]) => {
      const item = document.createElement('li');
      item.textContent = `${label}: ${NUMBER_FORMAT.format(value)}`;
      elements.metrics.appendChild(item);
    });
  }

  function applyFilters() {
    if (!currentCollections || !map.getLayer('atlas-lines-layer')) return;
    const gridVoltageFilter = voltageFilter();
    const minimumPlantMw = Number(elements.minimumPlantMw.value) || 0;
    map.setFilter('atlas-lines-layer', gridVoltageFilter);
    map.setFilter('atlas-substations-layer', gridVoltageFilter);
    map.setFilter('atlas-plants-layer', [
      '>=',
      ['coalesce', ['to-number', ['get', 'capacity_mw']], 0],
      minimumPlantMw,
    ]);
    setLayerVisibility('atlas-lines-layer', elements.showLines.checked);
    setLayerVisibility('atlas-substations-layer', elements.showSubstations.checked);
    setLayerVisibility('atlas-plants-layer', elements.showPlants.checked);
    setLayerVisibility('atlas-boundary-fill', elements.showBoundaries.checked);
    setLayerVisibility('atlas-boundary-line', elements.showBoundaries.checked);
    renderMetrics();
  }

  function popupLine(label, value) {
    if (!value && value !== 0) return null;
    const line = document.createElement('span');
    line.textContent = `${label}: ${value}`;
    return line;
  }

  function popupContent(properties) {
    const wrapper = document.createElement('div');
    wrapper.className = 'asset-popup';
    const heading = document.createElement('strong');
    heading.textContent = properties.name || properties.label || 'Grid feature';
    wrapper.appendChild(heading);

    const details = [];
    if (properties.asset_type) details.push(popupLine('Type', properties.asset_type));
    if (properties.voltage !== undefined) details.push(
      popupLine('Voltage', `${NUMBER_FORMAT.format(Number(properties.voltage))} kV`),
    );
    if (properties.capacity_mw !== undefined) details.push(
      popupLine('Capacity', `${NUMBER_FORMAT.format(Number(properties.capacity_mw))} MW`),
    );
    if (properties.category) details.push(popupLine('Category', properties.category));
    if (properties.owner) details.push(popupLine('Owner', properties.owner));
    if (properties.operator) details.push(popupLine('Operator', properties.operator));
    if (properties.status) details.push(popupLine('Status', properties.status));
    if (properties.place) details.push(popupLine('Location', properties.place));
    if (properties.source) details.push(popupLine('Source', properties.source));
    if (properties.voltage_source) {
      const confidence = properties.voltage_match_confidence === null
        ? ''
        : ` · ${Math.round(Number(properties.voltage_match_confidence) * 100)}% confidence`;
      const status = properties.voltage_match_status
        ? ` (${properties.voltage_match_status}${confidence})`
        : confidence;
      details.push(popupLine(
        'Voltage source',
        `${properties.voltage_source}${status}${
          properties.voltage_retrieved_at ? ` · retrieved ${properties.voltage_retrieved_at}` : ''
        }`,
      ));
    }
    details.filter(Boolean).forEach((detail) => wrapper.appendChild(detail));

    if (properties.source_url) {
      const sourceLink = document.createElement('a');
      sourceLink.href = properties.source_url;
      sourceLink.target = '_blank';
      sourceLink.rel = 'noopener noreferrer';
      sourceLink.textContent = 'Open public source';
      wrapper.appendChild(sourceLink);
    }
    if (properties.voltage_source_url) {
      const voltageSourceLink = document.createElement('a');
      voltageSourceLink.href = properties.voltage_source_url;
      voltageSourceLink.target = '_blank';
      voltageSourceLink.rel = 'noopener noreferrer';
      voltageSourceLink.textContent = 'Open voltage source';
      wrapper.appendChild(voltageSourceLink);
    }
    return wrapper;
  }

  function openPopup(coordinates, properties) {
    if (popup) popup.remove();
    popup = new window.maplibregl.Popup({ closeButton: true, closeOnClick: true })
      .setLngLat(coordinates)
      .setDOMContent(popupContent(properties))
      .addTo(map);
  }

  function interactiveLayers() {
    return ['atlas-plants-layer', 'atlas-substations-layer', 'atlas-lines-layer']
      .filter((id) => map.getLayer(id));
  }

  map.on('click', (event) => {
    const layers = interactiveLayers();
    if (!layers.length) return;
    const features = map.queryRenderedFeatures(event.point, { layers });
    const feature = features[0];
    if (!feature) return;
    openPopup([event.lngLat.lng, event.lngLat.lat], feature.properties || {});
  });

  map.on('mousemove', (event) => {
    const layers = interactiveLayers();
    if (!layers.length) return;
    const features = map.queryRenderedFeatures(event.point, { layers });
    map.getCanvas().style.cursor = features.length ? 'pointer' : '';
  });

  map.on('mouseout', () => {
    map.getCanvas().style.cursor = '';
  });

  async function responseJsonPossiblyGzip(response) {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} while opening ${response.url}`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    const isGzip = bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
    if (!isGzip) {
      return JSON.parse(new TextDecoder().decode(bytes));
    }
    if (!('DecompressionStream' in window)) {
      throw new Error('This browser cannot open compressed Atlas data. Please use a current browser.');
    }
    const decompressed = new Blob([bytes])
      .stream()
      .pipeThrough(new window.DecompressionStream('gzip'));
    return new Response(decompressed).json();
  }

  function validatePayload(payload, regionId) {
    if (!payload || payload.schema_version !== 1 || payload.region_id !== regionId) {
      throw new Error('The packaged Atlas shard did not pass its identity check.');
    }
    ['transmission_lines', 'substations', 'power_plants'].forEach((key) => {
      if (!Array.isArray(payload[key])) {
        throw new Error(`The packaged Atlas shard is missing ${key}.`);
      }
    });
    return payload;
  }

  function renderRegionButtons() {
    elements.regionButtons.replaceChildren();
    (manifest.regions || []).forEach((region) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'region-button';
      button.dataset.regionId = region.id;
      button.textContent = region.label;
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => {
        if (region.id !== currentRegion?.id) selectRegion(region.id);
      });
      elements.regionButtons.appendChild(button);
    });
  }

  function updateRegionButtons(regionId) {
    elements.regionButtons.querySelectorAll('.region-button').forEach((button) => {
      const active = button.dataset.regionId === regionId;
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
      button.disabled = active && !elements.loadingPanel.hidden;
    });
  }

  function updateUrl(regionId) {
    const next = new URL(window.location.href);
    next.searchParams.set('region', regionId);
    next.searchParams.delete('grid_region');
    window.history.replaceState({}, '', `${next.pathname}${next.search}${next.hash}`);
  }

  function updateSourceLink(payload) {
    const boundary = (payload.boundaries || []).find((item) => item.source_url);
    const sourceUrl = boundary?.source_url || manifest.sources?.iso_rto_boundaries;
    if (sourceUrl) {
      elements.regionSource.href = sourceUrl;
      elements.regionSource.hidden = false;
    } else {
      elements.regionSource.hidden = true;
    }
  }

  function fitRegion(region) {
    const bounds = Array.isArray(region.bounds) ? region.bounds.map(Number) : [];
    if (bounds.length === 4 && bounds.every(Number.isFinite)) {
      map.fitBounds(
        [[bounds[0], bounds[1]], [bounds[2], bounds[3]]],
        {
          padding: query.get('embed') === '1' ? 18 : 36,
          duration: 0,
          maxZoom: 7,
        },
      );
      return;
    }
    const center = region.center || {};
    map.jumpTo({
      center: [Number(center.lon) || -101, Number(center.lat) || 46],
      zoom: Number(region.zoom) || 2,
    });
  }

  function waitForMapIdle(timeoutMs = 15000) {
    return new Promise((resolve) => {
      let finished = false;
      const finish = () => {
        if (finished) return;
        finished = true;
        window.clearTimeout(timer);
        map.off('idle', finish);
        resolve();
      };
      const timer = window.setTimeout(finish, timeoutMs);
      map.once('idle', finish);
    });
  }

  async function selectRegion(regionId) {
    const region = regionsById.get(regionId) || regionsById.get(manifest.default_region);
    if (!region) {
      showFatalError('Grid area unavailable.', 'The selected area is not in the packaged manifest.');
      return;
    }

    const requestSequence = ++regionRequestSequence;
    if (activeRequest) activeRequest.abort();
    activeRequest = new AbortController();
    setLoading(
      `Opening packaged ${region.label} data…`,
      `Loading only the ${(Number(region.gzip_bytes) / 1048576).toFixed(1)} MB ${region.label} shard. No GIS, AI, or embedding request is running.`,
    );
    updateRegionButtons(region.id);

    try {
      const artifactUrl = new URL(region.artifact, new URL(MANIFEST_URL, window.location.href));
      const response = await fetch(artifactUrl, {
        signal: activeRequest.signal,
        cache: 'force-cache',
      });
      const payload = validatePayload(
        await responseJsonPossiblyGzip(response),
        region.id,
      );
      if (requestSequence !== regionRequestSequence) return;

      const collections = buildCollections(payload);
      await mapReady;
      if (requestSequence !== regionRequestSequence) return;
      currentRegion = region;
      currentCollections = collections;
      // Every grid opens unfiltered, including records whose voltage is unknown.
      // Keep this UI default independent of older cached manifest preferences.
      elements.minimumVoltage.value = '0';
      elements.minimumPlantMw.value = '0';
      elements.includeUnknownVoltage.checked = true;
      addAtlasData(collections, region);
      fitRegion(region);
      await waitForMapIdle();
      if (requestSequence !== regionRequestSequence) return;
      updateSourceLink(payload);
      updateUrl(region.id);
      updateRegionButtons(region.id);
      elements.status.textContent = `${region.label} ready from its saved ${(
        Number(region.gzip_bytes) / 1048576
      ).toFixed(1)} MB shard. Select another area only when needed.`;
      hideLoading();
    } catch (error) {
      if (error?.name === 'AbortError') return;
      console.error(error);
      showFatalError(
        `${region.label} could not be opened.`,
        `${error?.message || 'The packaged data is unavailable.'} Open the ERCOT dashboard as a fallback.`,
      );
    }
  }

  function showSearchResult(result, queryText) {
    const source = map.getSource('atlas-highlight');
    if (source) {
      source.setData({
        type: 'FeatureCollection',
        features: [{
          type: 'Feature',
          properties: {},
          geometry: result.geometry,
        }],
      });
    }
    map.easeTo({
      center: result.coordinates,
      zoom: Math.max(map.getZoom(), result.kind === 'point' ? 8 : 6),
      duration: prefersReducedMotion ? 0 : 650,
    });
    openPopup(result.coordinates, result.properties);
    elements.status.textContent = `Found “${result.label}” for “${queryText}”.`;
  }

  function searchAssets(event) {
    event.preventDefault();
    const queryText = elements.assetSearch.value.trim();
    const normalized = queryText.toLowerCase();
    if (!normalized) {
      elements.status.textContent = 'Enter a facility, line, place, fuel, voltage, owner, or status.';
      return;
    }
    const matches = searchIndex.filter((entry) => entry.text.includes(normalized));
    if (!matches.length) {
      elements.status.textContent = `No ${currentRegion?.label || 'regional'} record matched “${queryText}”. Try a shorter term or another grid area.`;
      return;
    }
    matches.sort((left, right) => {
      const leftPrefix = left.label.toLowerCase().startsWith(normalized) ? 0 : 1;
      const rightPrefix = right.label.toLowerCase().startsWith(normalized) ? 0 : 1;
      return leftPrefix - rightPrefix || left.label.localeCompare(right.label);
    });
    showSearchResult(matches[0], queryText);
    if (matches.length > 1) {
      elements.status.textContent += ` ${NUMBER_FORMAT.format(matches.length)} records matched; showing the closest name match.`;
    }
  }

  [
    elements.minimumVoltage,
    elements.minimumPlantMw,
    elements.showLines,
    elements.showSubstations,
    elements.showPlants,
    elements.showBoundaries,
    elements.includeUnknownVoltage,
  ].forEach((control) => control.addEventListener('change', applyFilters));
  elements.assetSearchForm.addEventListener('submit', searchAssets);

  async function initialize() {
    try {
      setLoading('Opening the saved Atlas manifest…', 'No live GIS request is running.');
      const [manifestResponse] = await Promise.all([
        fetch(MANIFEST_URL, { cache: 'no-store' }),
        mapReady,
      ]);
      if (!manifestResponse.ok) {
        throw new Error(`HTTP ${manifestResponse.status} while opening the Atlas manifest`);
      }
      manifest = await manifestResponse.json();
      if (!manifest || manifest.schema_version !== 1 || !Array.isArray(manifest.regions)) {
        throw new Error('The saved Atlas manifest is invalid.');
      }
      regionsById = new Map(manifest.regions.map((region) => [String(region.id), region]));
      elements.disclaimer.textContent = manifest.disclaimer || elements.disclaimer.textContent;
      renderRegionButtons();
      const requested = query.get('region') || query.get('grid_region') || manifest.default_region || 'ercot';
      await selectRegion(regionsById.has(requested) ? requested : manifest.default_region);
    } catch (error) {
      console.error(error);
      showFatalError(
        'The Grid Atlas could not start.',
        `${error?.message || 'The saved Atlas is unavailable.'} Open the ERCOT dashboard as a fallback.`,
      );
    }
  }

  initialize();
})();
