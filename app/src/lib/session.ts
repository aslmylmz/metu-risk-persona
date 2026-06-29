import type { GameEvent } from "./events";

/** The session payload POSTed to the scoring endpoint (mirrors scoring.schemas.GameSession). */
export interface SessionPayload {
  session_id: string;
  game_type: "BART_RISK";
  candidate_id: string;
  events: GameEvent[];
}

export function buildSessionPayload(
  sessionId: string,
  candidateId: string,
  events: GameEvent[],
): SessionPayload {
  return {
    session_id: sessionId,
    game_type: "BART_RISK",
    candidate_id: candidateId,
    events,
  };
}
