/**
 * Shared State Store — Project Vend.
 *
 * Central state that all agents read from and write to.
 * Prototype: in-memory → Production: SQLite → PostgreSQL.
 *
 * TODO: Implement state store.
 */

export interface Product {
  id: string;
  name: string;
  category: string;
  base_price: number;
  shelf_life_days: number | null;
}

export interface Slot {
  machine_id: string;
  slot_number: number;
  product_id: string;
  current_stock: number;
  max_capacity: number;
}

export interface Machine {
  id: string;
  location: string;
  status: "online" | "offline" | "maintenance";
  slots: Slot[];
}

export interface Transaction {
  id: string;
  machine_id: string;
  product_id: string;
  price_paid: number;
  timestamp: string;
  customer_session: string | null;
}

// TODO: Implement state store
// export class StateStore { ... }
