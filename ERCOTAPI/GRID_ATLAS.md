# U.S.–Canada Grid Atlas

The dashboard Grid Atlas is an offline-built public reference map. Streamlit
loads only checked-in gzip shards from `grid_atlas_data/`; opening the page or
changing regions does not bulk-download ArcGIS data, call OpenAI, or create
embeddings. The Atlas opens on ERCOT by default; the U.S.–Canada overview and
every other packaged regional filter remain available from the Grid area
selector. Records with an unknown voltage are visible by default in every
region and can still be hidden with the Unknown kV control.

The same saved shards power `grid-atlas.html`, the native interactive map
embedded on amirexirpe.com. That page lazy-loads ERCOT first and fetches a
different compressed shard only after the visitor selects another grid area;
it does not start Streamlit or preload every region.

## Packaged coverage

- `all`: performance-safe U.S.–Canada overview
  - U.S. transmission lines and substations at 230 kV and above
  - U.S. and Canadian plants at 100 MW and above
  - Canadian CanVec power lines and transformer stations
- Organized-market reference shards: ERCOT, MISO, PJM, CAISO, SPP, NYISO,
  and ISO-NE
- `canada`: national CanVec line/transformer geometry and NACEI plant records

The organized-market polygons are an approximate historical HIFLD reference.
Spatial location does not prove ISO/RTO membership, ownership, or electrical
connectivity. The displayed NERC reference polygons cover the contiguous
United States only.

## Sources and limitations

| Layer | Packaged source records | Important limitation |
|---|---:|---|
| U.S. transmission lines | 74,553 | Public HIFLD reference; source data last edited in 2022 |
| U.S. substations | 75,328 | Public HIFLD reference; item modified in 2021 |
| U.S. power plants | 13,446 | EIA layer, reporting period February 2025 |
| Canadian power lines | 13,009 | CanVec cartographic data; dates through September 2015; no voltage or owner |
| Canadian transformer features | 4,538 | CanVec point and polygon representations; no connectivity |
| Canadian plants | 1,125 | NACEI historical reference period August 2017 |

Every source URL, normalized count, input checksum, reporting date, regional
artifact checksum, compressed size, and record count is recorded in
`grid_atlas_data/manifest.json`.

## Offline refresh

Download and independently verify complete source extracts first. Then run:

```bash
python -m ERCOTAPI.grid_atlas_builder \
  --us-lines /path/to/transmission_full.geojson \
  --us-substations /path/to/substations_full.geojson \
  --us-plants /path/to/power_plants_full.geojson \
  --market-boundaries /path/to/Independent_System_Operators.shp \
  --nerc-boundaries /path/to/nerc_regions.geojson \
  --canvec-dir /path/to/canvec_50K_CA_Res_MGT \
  --canada-plants /path/to/nacei_canada_power_plants_100mw.geojson \
  --canada-plants /path/to/nacei_canada_renewable_plants_1mw.geojson \
  --output ERCOTAPI/grid_atlas_data
```

The builder:

1. rejects incomplete U.S. normalization or collapsed source counts;
2. removes HIFLD missing-value sentinels;
3. deduplicates overlapping Canadian plant layers;
4. tags U.S. assets against the original HIFLD ISO/RTO shapefile;
5. simplifies display geometry and creates one shard per filter;
6. writes deterministic gzip files into a staging directory; and
7. replaces the last good package only after every shard succeeds.

Run the Atlas tests before deployment:

```bash
python -m unittest \
  ERCOTAPI.tests.test_grid_atlas \
  ERCOTAPI.tests.test_grid_atlas_store \
  ERCOTAPI.tests.test_grid_atlas_app
```

The Atlas data are deliberately separate from the ERCOT engineering-document
RAG. GIS refreshes must never invoke the embedding pipeline.
