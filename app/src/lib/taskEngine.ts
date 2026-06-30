/** The pure BART task engine: gameplay transitions + burst resolution (SPEC §7.2).
 *
 * Stateless and side-effect-free — the heart of the Run task pulled out of the
 * React shell so the rules live in one tested place (and the test surface IS the
 * interface). Purity by inversion: a `pump` carries its random `draw`, and the
 * engine returns `events` without timestamps. The shell owns the rng,
 * `performance.now()`, and the feedback delay; it fires `next` after its timeout.
 */

import type { GameEvent } from "./events";

export type Phase = "playing" | "feedback" | "finished";
export type BalloonStatus = "active" | "collected" | "exploded";

/** Just what the engine needs about a balloon (a `run/sequence.ts` `Balloon` is
 * structurally assignable, so the engine carries no dependency on that module). */
export interface BalloonSpec {
  colorName: string;
  maxPumps: number;
  hazard: number[];
}

export interface BalloonResult {
  pumps: number;
  status: "collected" | "exploded";
}

export interface GameState {
  phase: Phase;
  index: number;
  pumps: number;
  status: BalloonStatus;
  completed: BalloonResult[];
  score: number;
}

export type Action =
  | { type: "pump"; draw: number }
  | { type: "collect" }
  | { type: "next" };

export interface EngineCtx {
  sequence: BalloonSpec[];
  reward: number;
}

/** An event the engine would log; the shell stamps it with `performance.now()`. */
export interface EngineEvent {
  type: GameEvent["type"];
  payload: Record<string, unknown>;
}

export interface AdvanceResult {
  state: GameState;
  events: EngineEvent[];
}

export function initialState(): GameState {
  return { phase: "playing", index: 0, pumps: 0, status: "active", completed: [], score: 0 };
}

const NOOP = (state: GameState): AdvanceResult => ({ state, events: [] });

/** Apply one action, returning the next state and any events to log. Invalid
 * actions for the current state (pumping a finished balloon, collecting nothing,
 * pumping past the cap) are no-ops. */
export function advance(state: GameState, action: Action, ctx: EngineCtx): AdvanceResult {
  const balloon = ctx.sequence[state.index];

  if (action.type === "pump") {
    if (state.phase !== "playing" || state.status !== "active") return NOOP(state);
    if (!balloon || state.pumps >= balloon.maxPumps) return NOOP(state);
    const newPumps = state.pumps + 1;
    const id = state.index + 1;
    const events: EngineEvent[] = [
      { type: "pump", payload: { balloon_id: id, color: balloon.colorName } },
    ];
    const hazard = balloon.hazard[newPumps - 1] ?? 1;
    if (action.draw < hazard) {
      events.push({
        type: "explode",
        payload: { balloon_id: id, color: balloon.colorName, pump_count: newPumps },
      });
      return { state: { ...state, pumps: newPumps, status: "exploded", phase: "feedback" }, events };
    }
    return { state: { ...state, pumps: newPumps }, events };
  }

  if (action.type === "collect") {
    if (state.phase !== "playing" || state.status !== "active" || state.pumps === 0) {
      return NOOP(state);
    }
    const id = state.index + 1;
    return {
      state: {
        ...state,
        status: "collected",
        phase: "feedback",
        score: state.score + state.pumps * ctx.reward,
      },
      events: [{ type: "collect", payload: { balloon_id: id, color: balloon.colorName } }],
    };
  }

  if (action.type === "next") {
    if (state.phase !== "feedback") return NOOP(state);
    const completed: BalloonResult[] = [
      ...state.completed,
      { pumps: state.pumps, status: state.status as "collected" | "exploded" },
    ];
    if (completed.length >= ctx.sequence.length) {
      return { state: { ...state, completed, phase: "finished" }, events: [] };
    }
    return {
      state: {
        phase: "playing",
        index: state.index + 1,
        pumps: 0,
        status: "active",
        completed,
        score: state.score,
      },
      events: [],
    };
  }

  return NOOP(state);
}
