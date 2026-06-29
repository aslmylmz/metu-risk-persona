export type ColorName = "orange" | "teal" | "purple";

/** Participant-facing balloon hexes -> the color names the scoring engine expects. */
const HEX_TO_COLOR: Record<string, ColorName> = {
  "#F97316": "orange",
  "#14B8A6": "teal",
  "#A855F7": "purple",
};

export function colorNameFromHex(hex: string): ColorName {
  return HEX_TO_COLOR[hex.toUpperCase()] ?? "teal";
}
