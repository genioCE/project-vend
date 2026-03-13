/**
 * Simulation Engine — Project Vend.
 *
 * Generates synthetic vending machine events for testing the gravity
 * orchestration pipeline without physical hardware.
 *
 * Modes:
 * - Real-time: 1:1 time progression
 * - Accelerated: Configurable speed (10x, 100x)
 * - Scripted: Predefined scenario sequences
 * - Random: Stochastic customer and event generation
 *
 * TODO: Implement simulation logic.
 */

export interface SimulationConfig {
  mode: "realtime" | "accelerated" | "scripted" | "random";
  speed_multiplier: number;
  product_count: number;
  machine_id: string;
  customer_arrival_rate: number; // per hour (Poisson)
}

export const DEFAULT_CONFIG: SimulationConfig = {
  mode: "accelerated",
  speed_multiplier: 10,
  product_count: 15,
  machine_id: "uco-main-001",
  customer_arrival_rate: 30,
};

// TODO: Implement simulation engine
// export class SimulationEngine { ... }
