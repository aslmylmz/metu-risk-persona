import { describe, expect, it } from "vitest";

import { colorNameFromHex } from "./colors";

describe("colorNameFromHex", () => {
  it("maps the three task hexes to engine color names", () => {
    expect(colorNameFromHex("#F97316")).toBe("orange");
    expect(colorNameFromHex("#14B8A6")).toBe("teal");
    expect(colorNameFromHex("#A855F7")).toBe("purple");
  });

  it("is case-insensitive on the hex", () => {
    expect(colorNameFromHex("#a855f7")).toBe("purple");
  });

  it("falls back to teal for an unknown hex", () => {
    expect(colorNameFromHex("#000000")).toBe("teal");
  });
});
