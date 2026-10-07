# True American League playoff odds

Power rankings and playoff odds for a Sleeper fantasy football league.

- `build.py` downloads the league from Sleeper's public API, checks totals against Sleeper's standings, simulates the rest of the season for every finished week, and writes `site/index.html`.
- `template.html` is the page.
- `.github/workflows/update.yml` rebuilds and publishes to GitHub Pages every Tuesday at 14:00 UTC, on every push to `main`, and on demand (Actions tab → Update playoff odds → Run workflow).

Run locally: `pip install numpy` then `python build.py`.
