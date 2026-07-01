import { describe, expect, it } from "vitest";

import { HAZARD_FAMILIES } from "../lib/config";
import { defaultHazard } from "./familyParams";

describe("defaultHazard", () => {
  it("produces a spec tagged with the requested family for every family", () => {
    for (const family of HAZARD_FAMILIES) {
      const spec = defaultHazard(family, 10);
      expect(spec.family).toBe(family);
    }
  });

  it("adds the correct scalar defaults", () => {
    // logistic is a good test case: it has 3 scalar params
    const spec = defaultHazard("logistic", 10) as any;
    expect(spec.family).toBe("logistic");
    expect(spec.h_max).toBe(0.9);
    expect(spec.midpoint).toBe(16);
    expect(spec.steepness).toBe(0.3);
  });

  it("handles empty scalar params correctly", () => {
    const spec = defaultHazard("dynamic", 10);
    expect(spec).toEqual({ family: "dynamic" });
    
    const spec2 = defaultHazard("lejuez", 10);
    expect(spec2).toEqual({ family: "lejuez" });
  });

  it("seeds the array families well-formed against maxPumps", () => {
    const step = defaultHazard("step", 32);
    if (step.family !== "step") throw new Error("wrong family");
    expect(step.levels.length).toBe(step.breakpoints.length + 1);
    expect(step.breakpoints[0]).toBeGreaterThanOrEqual(1);

    const tabular = defaultHazard("tabular", 8);
    if (tabular.family !== "tabular") throw new Error("wrong family");
    expect(tabular.values).toHaveLength(8);
    expect(tabular.values.every((v) => v >= 0 && v <= 1)).toBe(true);
  });
});
