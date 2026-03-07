import duckdb from "duckdb";
import { existsSync, mkdirSync } from "fs";
import { dirname } from "path";

const DUCKDB_PATH =
  process.env.DUCKDB_PATH || "./data/timeseries.duckdb";

let _db: duckdb.Database | null = null;

function getDb(): duckdb.Database {
  if (!_db) {
    const dir = dirname(DUCKDB_PATH);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
    _db = new duckdb.Database(DUCKDB_PATH);
  }
  return _db;
}

function coerceBigInts(row: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(row)) {
    out[k] = typeof v === "bigint" ? Number(v) : v;
  }
  return out;
}

export function query(
  sql: string,
  params: unknown[] = []
): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    const db = getDb();
    const cb = (err: Error | null, rows: Record<string, unknown>[]) => {
      if (err) reject(err);
      else resolve((rows ?? []).map(coerceBigInts));
    };
    const args: unknown[] = [sql, ...params, cb];
    (db.all as Function).apply(db, args);
  });
}

export function run(
  sql: string,
  params: unknown[] = []
): Promise<void> {
  return new Promise((resolve, reject) => {
    const db = getDb();
    const cb = (err: Error | null) => {
      if (err) reject(err);
      else resolve();
    };
    const args: unknown[] = [sql, ...params, cb];
    (db.run as Function).apply(db, args);
  });
}

export function exec(sql: string): Promise<void> {
  return new Promise((resolve, reject) => {
    getDb().exec(sql, (err: Error | null) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

export function close(): Promise<void> {
  return new Promise((resolve, reject) => {
    if (_db) {
      _db.close((err: Error | null) => {
        _db = null;
        if (err) reject(err);
        else resolve();
      });
    } else {
      resolve();
    }
  });
}
