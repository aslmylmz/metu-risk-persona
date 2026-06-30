import { describe, expect, it } from "vitest";

import {
  advance,
  type EngineCtx,
  type EngineEvent,
  type GameState,
  initialState,
} from "./taskEngine";

// A one-balloon context: purple, cap 3, never bursts on pumps 1–2, certain at 3.
const ctx = (overrides: Partial<EngineCtx> = {}): EngineCtx => ({
  sequence: [{ colorName: "purple", maxPumps: 3, hazard: [0, 0, 1] }],
  reward: 0.25,
  ...overrides,
});

describe("initialState", () => {
  it("starts playing on the first balloon", () => {
    expect(initialState()).toEqual({
      phase: "playing",
      index: 0,
      pumps: 0,
      status: "active",
      completed: [],
      score: 0,
    });
  });
});

describe("advance · pump", () => {
  it("a surviving pump increments pumps and logs one pump event", () => {
    const { state, events } = advance(initialState(), { type: "pump", draw: 0.9 }, ctx());
    expect(state).toMatchObject({ pumps: 1, status: "active", phase: "playing" });
    expect(events).toEqual([{ type: "pump", payload: { balloon_id: 1, color: "purple" } }]);
  });

  it("a bursting pump explodes the balloon and logs pump + explode(pump_count)", () => {
    const atTwo: GameState = { ...initialState(), pumps: 2 };
    const { state, events } = advance(atTwo, { type: "pump", draw: 0.5 }, ctx()); // hazard[2] = 1
    expect(state).toMatchObject({ pumps: 3, status: "exploded", phase: "feedback" });
    expect(events).toEqual([
      { type: "pump", payload: { balloon_id: 1, color: "purple" } },
      { type: "explode", payload: { balloon_id: 1, color: "purple", pump_count: 3 } },
    ]);
  });

  it("bursts iff draw < hazard[k-1]", () => {
    const c = ctx({ sequence: [{ colorName: "teal", maxPumps: 2, hazard: [0.5, 1] }] });
    expect(advance(initialState(), { type: "pump", draw: 0.49 }, c).state.status).toBe("exploded");
    expect(advance(initialState(), { type: "pump", draw: 0.5 }, c).state.status).toBe("active");
  });
});

describe("advance · collect", () => {
  it("banks pumps × reward, marks collected, and logs a collect event", () => {
    const atFour: GameState = { ...initialState(), pumps: 4, score: 1.0 };
    const { state, events } = advance(atFour, { type: "collect" }, ctx({ reward: 0.25 }));
    expect(state).toMatchObject({ status: "collected", phase: "feedback", score: 2.0 });
    expect(events).toEqual([{ type: "collect", payload: { balloon_id: 1, color: "purple" } }]);
  });

  it("is a no-op when nothing has been pumped", () => {
    const result = advance(initialState(), { type: "collect" }, ctx());
    expect(result.events).toEqual([]);
    expect(result.state).toEqual(initialState());
  });
});

describe("advance · next", () => {
  const twoBalloons = ctx({
    sequence: [
      { colorName: "purple", maxPumps: 3, hazard: [0, 0, 1] },
      { colorName: "teal", maxPumps: 2, hazard: [0, 1] },
    ],
  });

  it("records the finished balloon and moves to the next, fresh", () => {
    const done: GameState = { ...initialState(), pumps: 2, status: "collected", phase: "feedback" };
    const { state, events } = advance(done, { type: "next" }, twoBalloons);
    expect(events).toEqual([]);
    expect(state).toEqual({
      phase: "playing",
      index: 1,
      pumps: 0,
      status: "active",
      completed: [{ pumps: 2, status: "collected" }],
      score: 0,
    });
  });

  it("finishes after the last balloon", () => {
    const lastDone: GameState = {
      phase: "feedback",
      index: 1,
      pumps: 1,
      status: "exploded",
      completed: [{ pumps: 2, status: "collected" }],
      score: 0.5,
    };
    const { state } = advance(lastDone, { type: "next" }, twoBalloons);
    expect(state.phase).toBe("finished");
    expect(state.completed).toEqual([
      { pumps: 2, status: "collected" },
      { pumps: 1, status: "exploded" },
    ]);
  });

  it("is a no-op outside the feedback phase", () => {
    expect(advance(initialState(), { type: "next" }, twoBalloons).state).toEqual(initialState());
  });
});

describe("advance · guards", () => {
  it("ignores a pump at the cap (cap precedes the hazard draw)", () => {
    const atCap: GameState = { ...initialState(), pumps: 3 }; // cap is 3
    const result = advance(atCap, { type: "pump", draw: 0 }, ctx());
    expect(result.events).toEqual([]);
    expect(result.state.pumps).toBe(3);
  });

  it("ignores a pump once the balloon is no longer active", () => {
    const popped: GameState = { ...initialState(), status: "exploded", phase: "feedback", pumps: 3 };
    expect(advance(popped, { type: "pump", draw: 0 }, ctx())).toEqual({ state: popped, events: [] });
  });
});

describe("advance · full playthrough", () => {
  it("plays a balloon pump-by-pump, then banks it", () => {
    const c = ctx({
      sequence: [{ colorName: "purple", maxPumps: 5, hazard: [0, 0, 0, 0, 1] }],
      reward: 0.5,
    });
    let s = initialState();
    const log: EngineEvent[] = [];
    for (let k = 1; k <= 4; k++) {
      const r = advance(s, { type: "pump", draw: 0.99 }, c);
      s = r.state;
      log.push(...r.events);
      expect(s.pumps).toBe(k);
      expect(s.status).toBe("active");
    }
    const collected = advance(s, { type: "collect" }, c);
    s = collected.state;
    log.push(...collected.events);

    expect(s).toMatchObject({ status: "collected", phase: "feedback", score: 2.0 });
    expect(log.filter((e) => e.type === "pump")).toHaveLength(4);
    expect(log.filter((e) => e.type === "collect")).toHaveLength(1);
    expect(advance(s, { type: "next" }, c).state.phase).toBe("finished");
  });
});
