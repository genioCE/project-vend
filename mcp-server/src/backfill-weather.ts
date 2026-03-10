#!/usr/bin/env npx ts-node
/**
 * Backfill weather data for all journal entries using Open-Meteo Historical API.
 *
 * Location: Oklahoma City, OK (73106/73112 area)
 * Coordinates: 35.47, -97.52
 *
 * Usage:
 *   npx ts-node scripts/backfill-weather.ts
 *   # or after build:
 *   node dist/scripts/backfill-weather.js
 */

import { query, exec, run } from "./timeseries/db.js";
import { ensureSchema } from "./timeseries/schema.js";

const LAT = 35.47;
const LON = -97.52;
const TIMEZONE = "America/Chicago";

// Open-Meteo Historical Weather API
const BASE_URL = "https://archive-api.open-meteo.com/v1/archive";

// WMO Weather Codes → Human readable conditions
const WMO_CODES: Record<number, string> = {
  0: "clear",
  1: "mainly_clear",
  2: "partly_cloudy",
  3: "overcast",
  45: "fog",
  48: "freezing_fog",
  51: "light_drizzle",
  53: "moderate_drizzle",
  55: "dense_drizzle",
  56: "freezing_drizzle",
  57: "dense_freezing_drizzle",
  61: "slight_rain",
  63: "moderate_rain",
  65: "heavy_rain",
  66: "freezing_rain",
  67: "heavy_freezing_rain",
  71: "slight_snow",
  73: "moderate_snow",
  75: "heavy_snow",
  77: "snow_grains",
  80: "slight_rain_showers",
  81: "moderate_rain_showers",
  82: "violent_rain_showers",
  85: "slight_snow_showers",
  86: "heavy_snow_showers",
  95: "thunderstorm",
  96: "thunderstorm_with_hail",
  99: "thunderstorm_with_heavy_hail",
};

interface DailyWeather {
  date: string;
  temperature_max: number;
  temperature_min: number;
  temperature_mean: number;
  apparent_temperature_max: number;
  apparent_temperature_min: number;
  precipitation_sum: number;
  rain_sum: number;
  snowfall_sum: number;
  weather_code: number;
  weather_condition: string;
  cloud_cover_mean: number;
  wind_speed_max: number;
  sunrise: string;
  sunset: string;
  daylight_hours: number;
}

async function fetchWeatherBatch(
  startDate: string,
  endDate: string
): Promise<DailyWeather[]> {
  const params = new URLSearchParams({
    latitude: LAT.toString(),
    longitude: LON.toString(),
    start_date: startDate,
    end_date: endDate,
    timezone: TIMEZONE,
    daily: [
      "temperature_2m_max",
      "temperature_2m_min",
      "temperature_2m_mean",
      "apparent_temperature_max",
      "apparent_temperature_min",
      "precipitation_sum",
      "rain_sum",
      "snowfall_sum",
      "weather_code",
      "cloud_cover_mean",
      "wind_speed_10m_max",
      "sunrise",
      "sunset",
    ].join(","),
  });

  const url = `${BASE_URL}?${params}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Open-Meteo API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const daily = data.daily;

  const results: DailyWeather[] = [];
  for (let i = 0; i < daily.time.length; i++) {
    const sunrise = daily.sunrise[i];
    const sunset = daily.sunset[i];

    // Calculate daylight hours
    let daylightHours = 0;
    if (sunrise && sunset) {
      const sunriseTime = new Date(sunrise).getTime();
      const sunsetTime = new Date(sunset).getTime();
      daylightHours = (sunsetTime - sunriseTime) / (1000 * 60 * 60);
    }

    const weatherCode = daily.weather_code[i] ?? 0;

    results.push({
      date: daily.time[i],
      temperature_max: daily.temperature_2m_max[i],
      temperature_min: daily.temperature_2m_min[i],
      temperature_mean: daily.temperature_2m_mean[i],
      apparent_temperature_max: daily.apparent_temperature_max[i],
      apparent_temperature_min: daily.apparent_temperature_min[i],
      precipitation_sum: daily.precipitation_sum[i] ?? 0,
      rain_sum: daily.rain_sum[i] ?? 0,
      snowfall_sum: daily.snowfall_sum[i] ?? 0,
      weather_code: weatherCode,
      weather_condition: WMO_CODES[weatherCode] ?? "unknown",
      cloud_cover_mean: daily.cloud_cover_mean[i] ?? 0,
      wind_speed_max: daily.wind_speed_10m_max[i] ?? 0,
      sunrise: sunrise?.split("T")[1] ?? "",
      sunset: sunset?.split("T")[1] ?? "",
      daylight_hours: Math.round(daylightHours * 100) / 100,
    });
  }

  return results;
}

async function addWeatherColumns(): Promise<void> {
  const columns = [
    "weather_temp_max DOUBLE",
    "weather_temp_min DOUBLE",
    "weather_temp_mean DOUBLE",
    "weather_temp_feels_max DOUBLE",
    "weather_temp_feels_min DOUBLE",
    "weather_precip_mm DOUBLE",
    "weather_rain_mm DOUBLE",
    "weather_snow_cm DOUBLE",
    "weather_code INTEGER",
    "weather_condition TEXT",
    "weather_cloud_cover DOUBLE",
    "weather_wind_max DOUBLE",
    "weather_sunrise TEXT",
    "weather_sunset TEXT",
    "weather_daylight_hours DOUBLE",
  ];

  for (const col of columns) {
    const [name] = col.split(" ");
    try {
      await exec(`ALTER TABLE entries ADD COLUMN ${col}`);
      console.log(`  Added column: ${name}`);
    } catch {
      // Column already exists
    }
  }
}

async function main(): Promise<void> {
  console.log("=== Weather Backfill for Journal Entries ===\n");
  console.log(`Location: Oklahoma City, OK (${LAT}, ${LON})`);
  console.log(`Timezone: ${TIMEZONE}\n`);

  // Ensure schema exists
  await ensureSchema();

  // Add weather columns
  console.log("Adding weather columns to entries table...");
  await addWeatherColumns();
  console.log();

  // Get all unique entry dates
  const entries = await query(
    `SELECT DISTINCT entry_date::TEXT AS date FROM entries ORDER BY entry_date`
  );

  if (entries.length === 0) {
    console.log("No entries found in database.");
    return;
  }

  console.log(`Found ${entries.length} unique entry dates\n`);

  const startDate = entries[0].date as string;
  const endDate = entries[entries.length - 1].date as string;

  console.log(`Date range: ${startDate} to ${endDate}\n`);

  // Fetch weather in batches (Open-Meteo handles large ranges well)
  // But let's do it in yearly batches to be safe
  const startYear = parseInt(startDate.slice(0, 4));
  const endYear = parseInt(endDate.slice(0, 4));

  const allWeather: Map<string, DailyWeather> = new Map();

  for (let year = startYear; year <= endYear; year++) {
    const batchStart = year === startYear ? startDate : `${year}-01-01`;
    const batchEnd = year === endYear ? endDate : `${year}-12-31`;

    console.log(`Fetching weather for ${year}...`);

    try {
      const weather = await fetchWeatherBatch(batchStart, batchEnd);
      for (const w of weather) {
        allWeather.set(w.date, w);
      }
      console.log(`  Got ${weather.length} days`);

      // Rate limiting - be nice to the free API
      await new Promise((r) => setTimeout(r, 500));
    } catch (err) {
      console.error(`  Error fetching ${year}: ${err}`);
    }
  }

  console.log(`\nTotal weather records: ${allWeather.size}`);

  // Update entries with weather data
  console.log("\nUpdating entries with weather data...");

  let updated = 0;
  let missing = 0;

  for (const entry of entries) {
    const date = entry.date as string;
    const weather = allWeather.get(date);

    if (!weather) {
      missing++;
      continue;
    }

    await run(
      `UPDATE entries SET
        weather_temp_max = ?,
        weather_temp_min = ?,
        weather_temp_mean = ?,
        weather_temp_feels_max = ?,
        weather_temp_feels_min = ?,
        weather_precip_mm = ?,
        weather_rain_mm = ?,
        weather_snow_cm = ?,
        weather_code = ?,
        weather_condition = ?,
        weather_cloud_cover = ?,
        weather_wind_max = ?,
        weather_sunrise = ?,
        weather_sunset = ?,
        weather_daylight_hours = ?
      WHERE entry_date = ?`,
      [
        weather.temperature_max,
        weather.temperature_min,
        weather.temperature_mean,
        weather.apparent_temperature_max,
        weather.apparent_temperature_min,
        weather.precipitation_sum,
        weather.rain_sum,
        weather.snowfall_sum / 10, // Convert mm to cm
        weather.weather_code,
        weather.weather_condition,
        weather.cloud_cover_mean,
        weather.wind_speed_max,
        weather.sunrise,
        weather.sunset,
        weather.daylight_hours,
        date,
      ]
    );

    updated++;
    if (updated % 100 === 0) {
      console.log(`  Updated ${updated} entries...`);
    }
  }

  console.log(`\nDone!`);
  console.log(`  Updated: ${updated} entries`);
  console.log(`  Missing weather data: ${missing} entries`);

  // Show sample
  console.log("\nSample weather data:");
  const sample = await query(
    `SELECT entry_date::TEXT AS date, weather_temp_max, weather_temp_min,
            weather_condition, weather_cloud_cover, weather_daylight_hours
     FROM entries
     WHERE weather_temp_max IS NOT NULL
     ORDER BY entry_date DESC
     LIMIT 5`
  );
  console.table(sample);
}

main().catch(console.error);
