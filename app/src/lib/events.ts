/** A single timestamped event logged during the task (mirrors scoring.schemas.GameEvent). */
export interface GameEvent {
  timestamp: number;
  type: "pump" | "collect" | "explode";
  payload: Record<string, unknown>;
}
