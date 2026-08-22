CREATE TABLE IF NOT EXISTS site_totals (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  total_views INTEGER NOT NULL DEFAULT 0 CHECK (total_views >= 0),
  updated_at TEXT
);

INSERT OR IGNORE INTO site_totals (id, total_views, updated_at)
VALUES (1, 0, NULL);

CREATE TABLE IF NOT EXISTS daily_totals (
  day TEXT PRIMARY KEY,
  pageviews INTEGER NOT NULL DEFAULT 0 CHECK (pageviews >= 0)
);

CREATE TABLE IF NOT EXISTS country_totals (
  country TEXT PRIMARY KEY CHECK (length(country) = 2),
  views INTEGER NOT NULL DEFAULT 0 CHECK (views >= 0)
);

CREATE TABLE IF NOT EXISTS visitor_days (
  day TEXT NOT NULL,
  visitor_hash TEXT NOT NULL CHECK (length(visitor_hash) = 64),
  country TEXT CHECK (country IS NULL OR length(country) = 2),
  counted_views INTEGER NOT NULL DEFAULT 0 CHECK (counted_views >= 0),
  last_seen_at TEXT NOT NULL,
  PRIMARY KEY (day, visitor_hash)
);

CREATE INDEX IF NOT EXISTS visitor_days_day_idx ON visitor_days(day);
CREATE INDEX IF NOT EXISTS country_totals_views_idx ON country_totals(views DESC);
