import BartGame from "./BartGame";

// Minimal Run wrapper. The full consent -> participant-ID -> debrief flow is Phase 3;
// for now we render the decoupled task directly.
export function App() {
  return <BartGame candidateId="anonymous" />;
}
