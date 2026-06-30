import { useEffect, useRef, useState } from "react";

import { persistSession, submitSession } from "./lib/api";
import type { TaskConfig } from "./lib/config";
import type { GameEvent } from "./lib/events";
import { taskStrings } from "./lib/i18n";
import { buildSessionPayload } from "./lib/session";
import {
  advance,
  type EngineCtx,
  type EngineEvent,
  type GameState,
  initialState,
} from "./lib/taskEngine";
import { type AssessmentResult, Debrief } from "./run/Debrief";
import { type Balloon, buildSequence, mulberry32 } from "./run/sequence";

// ── Types ───────────────────────────────────────────────────────────────────

interface BalloonState {
    id: number;
    pumps: number;
    status: "active" | "collected" | "exploded";
}

// ── Component ───────────────────────────────────────────────────────────────
//
// A thin rendering shell over the pure task engine (lib/taskEngine.ts). The engine
// owns the gameplay rules; this component owns the seeded rng, timestamps, the
// feedback delay, and the view. It dispatches user input to `advance()` and derives
// the view names (gamePhase, currentBalloon, …) from the engine state so the markup
// stays declarative. The results screen is the standalone <Debrief>.

interface BartGameProps {
    config: TaskConfig;
    hazards: Record<string, number[]>;
    candidateId: string;
    onComplete?: (data: AssessmentResult) => void;
}

export default function BartGame({ config, hazards, candidateId, onComplete }: BartGameProps) {
    const eventLogRef = useRef<GameEvent[]>([]);
    const sessionIdRef = useRef(crypto.randomUUID());
    const sequenceRef = useRef<Balloon[]>([]);
    const rngRef = useRef<() => number>(() => Math.random());
    const feedbackTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
    const containerRef = useRef<HTMLDivElement>(null);

    const t = taskStrings(config.language);
    const totalBalloons = config.colors.reduce((n, c) => n + c.trials, 0);

    const [engine, setEngine] = useState<GameState>(initialState);
    const [started, setStarted] = useState(false);
    const [feedbackMessage, setFeedbackMessage] = useState("");
    const [results, setResults] = useState<AssessmentResult | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const ctx = (): EngineCtx => ({ sequence: sequenceRef.current, reward: config.reward_per_pump });

    /** Stamp the engine's events with a monotonic timestamp and append to the log. */
    const logEvents = (events: EngineEvent[]) => {
        const ts = performance.now();
        for (const e of events) {
            eventLogRef.current.push({ timestamp: ts, type: e.type, payload: e.payload });
        }
    };

    /** After feedback, advance to the next balloon (or finish) — the timing the
     * engine deliberately leaves to the view. */
    const scheduleNext = (delay: number) => {
        feedbackTimer.current = setTimeout(() => {
            setEngine((s) => advance(s, { type: "next" }, ctx()).state);
            setFeedbackMessage("");
        }, delay);
    };

    const startGame = () => {
        eventLogRef.current = [];
        sessionIdRef.current = crypto.randomUUID();
        // One seeded rng drives both the shuffle and the per-pump burst draws, so a
        // fixed seed reproduces the whole run (SPEC §7.2); null seed → fresh run.
        const seed = config.seed ?? ((Math.random() * 2 ** 32) >>> 0);
        const rng = mulberry32(seed);
        rngRef.current = rng;
        sequenceRef.current = buildSequence(config, hazards, rng);
        setEngine(initialState());
        setResults(null);
        setFeedbackMessage("");
        setStarted(true);
    };

    const handlePump = () => {
        if (engine.phase !== "playing" || engine.status !== "active") return;
        const { state, events } = advance(engine, { type: "pump", draw: rngRef.current() }, ctx());
        logEvents(events);
        setEngine(state);
        if (state.status === "exploded") {
            setFeedbackMessage(t.exploded);
            scheduleNext(1200);
        }
    };

    const handleCollect = () => {
        if (engine.phase !== "playing" || engine.status !== "active" || engine.pumps === 0) return;
        const money = engine.pumps * config.reward_per_pump;
        const { state, events } = advance(engine, { type: "collect" }, ctx());
        logEvents(events);
        setEngine(state);
        setFeedbackMessage(`${t.collected} $${money.toFixed(2)}`);
        scheduleNext(1000);
    };

    const handleSubmit = async () => {
        setIsSubmitting(true);
        const payload = buildSessionPayload(sessionIdRef.current, candidateId, eventLogRef.current);
        try {
            const data = await submitSession<AssessmentResult>(payload, config);
            setResults(data);

            // Persist the session locally via the sidecar (best-effort; the engine
            // owns file writing, SPEC §13). A write failure must not block results.
            void persistSession(payload, config).catch((persistErr) =>
                console.error("Failed to persist session:", persistErr),
            );

            if (onComplete) onComplete(data);
        } catch (err) {
            console.error("Submission error:", err);
            setFeedbackMessage(err instanceof Error ? err.message : "Failed to submit");
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.code === "Space") {
            e.preventDefault();
            handlePump();
        } else if (e.code === "Enter") {
            e.preventDefault();
            handleCollect();
        }
    };

    // Derived view state: map the engine onto the names the markup already uses.
    const gamePhase: "idle" | "playing" | "feedback" | "finished" | "results" = results
        ? "results"
        : !started
            ? "idle"
            : engine.phase;
    const currentBalloon: BalloonState = {
        id: engine.index + 1,
        pumps: engine.pumps,
        status: engine.status,
    };
    const completedBalloons: BalloonState[] = engine.completed.map((c, i) => ({
        id: i + 1,
        pumps: c.pumps,
        status: c.status,
    }));
    const totalScore = engine.score;
    const balloonCount = engine.completed.length + 1;

    useEffect(() => {
        if (gamePhase === "playing" && containerRef.current) {
            containerRef.current.focus();
        }
    }, [gamePhase]);

    // Clear any pending feedback timer on unmount.
    useEffect(() => () => {
        if (feedbackTimer.current) clearTimeout(feedbackTimer.current);
    }, []);

    const currentConfig = sequenceRef.current[engine.index];
    const balloonColor = currentConfig ? currentConfig.displayHex : "#9CA3AF";
    const balloonScale = 1 + currentBalloon.pumps * 0.08;
    const balloonSize = 100 * balloonScale;

    return (
        <div
            ref={containerRef}
            className="w-full"
            onKeyDown={handleKeyDown}
            tabIndex={0}
            style={{ outline: "none" }}
        >
            {/* ── Idle Screen ──────────────────────────────────────────────────── */}
            {gamePhase === "idle" && (
                <div className="flex flex-col items-center py-16 gap-6">
                    <div className="text-6xl">🎈</div>
                    <h2
                        style={{
                            fontSize: "1.5rem",
                            fontWeight: 700,
                            color: "#fff",
                        }}
                    >
                        {t.taskTitle}
                    </h2>
                    <p
                        style={{
                            color: "#9CA3AF",
                            textAlign: "center",
                            maxWidth: "400px",
                            lineHeight: 1.6,
                        }}
                    >
                        {t.instructions}
                    </p>
                    <p
                        style={{ color: "#6B7280", fontSize: "0.85rem" }}
                    >
                        {totalBalloons} {t.balloonsWord} · {t.controlsHint}
                    </p>
                    <button
                        onClick={startGame}
                        style={{
                            marginTop: "0.5rem",
                            padding: "12px 40px",
                            fontSize: "1rem",
                            fontWeight: 600,
                            color: "#fff",
                            background: "linear-gradient(135deg, #6366F1, #8B5CF6)",
                            border: "none",
                            borderRadius: "12px",
                            cursor: "pointer",
                            transition: "transform 0.15s, box-shadow 0.15s",
                            boxShadow: "0 4px 15px rgba(99,102,241,0.4)",
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.transform = "translateY(-2px)";
                            e.currentTarget.style.boxShadow =
                                "0 6px 20px rgba(99,102,241,0.6)";
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.transform = "translateY(0)";
                            e.currentTarget.style.boxShadow =
                                "0 4px 15px rgba(99,102,241,0.4)";
                        }}
                    >
                        {t.startButton}
                    </button>
                </div>
            )}

            {/* ── Playing / Feedback ───────────────────────────────────────────── */}
            {(gamePhase === "playing" || gamePhase === "feedback") && (
                <div className="flex flex-col items-center py-8 gap-6">
                    <div
                        style={{
                            display: "flex",
                            justifyContent: "space-between",
                            width: "100%",
                            maxWidth: 400,
                            padding: "0 1rem",
                        }}
                    >
                        <span style={{ color: "#9CA3AF", fontSize: "0.9rem" }}>
                            {t.balloonLabel}{" "}
                            <span style={{ color: "#fff", fontWeight: 700 }}>
                                {Math.min(balloonCount, totalBalloons)}
                            </span>
                            /{totalBalloons}
                        </span>
                        <span style={{ color: "#9CA3AF", fontSize: "0.9rem" }}>
                            {t.totalLabel}{" "}
                            <span style={{ color: "#22C55E", fontWeight: 700 }}>
                                ${totalScore.toFixed(2)}
                            </span>
                        </span>
                    </div>

                    <div
                        style={{
                            position: "relative",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            height: "280px",
                        }}
                    >
                        {currentBalloon.status === "exploded" ? (
                            <div
                                style={{
                                    fontSize: "5rem",
                                    animation: "fadeIn 0.2s ease-out",
                                }}
                            >
                                💥
                            </div>
                        ) : (
                            <div
                                style={{
                                    width: `${balloonSize}px`,
                                    height: `${balloonSize * 1.2}px`,
                                    borderRadius: "50% 50% 50% 50% / 40% 40% 60% 60%",
                                    background: `radial-gradient(circle at 35% 35%, ${balloonColor}CC, ${balloonColor})`,
                                    boxShadow: `0 8px 30px ${balloonColor}40, inset 0 -8px 20px rgba(0,0,0,0.2)`,
                                    transition: "width 0.15s ease, height 0.15s ease",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    position: "relative",
                                }}
                            >
                                <div
                                    style={{
                                        position: "absolute",
                                        top: "20%",
                                        left: "30%",
                                        width: "25%",
                                        height: "20%",
                                        borderRadius: "50%",
                                        background:
                                            "radial-gradient(ellipse, rgba(255,255,255,0.4), transparent)",
                                    }}
                                />
                                <span
                                    style={{
                                        fontSize: `${Math.max(1.2, 1.8 * balloonScale * 0.4)}rem`,
                                        fontWeight: 800,
                                        color: "rgba(255,255,255,0.9)",
                                        textShadow: "0 2px 4px rgba(0,0,0,0.3)",
                                    }}
                                >
                                    ${(currentBalloon.pumps * config.reward_per_pump).toFixed(2)}
                                </span>
                            </div>
                        )}

                        {currentBalloon.status === "active" && (
                            <div
                                style={{
                                    position: "absolute",
                                    bottom: `${-20 + (280 - balloonSize * 1.2) / 2}px`,
                                    width: "2px",
                                    height: "30px",
                                    background: "#6B7280",
                                }}
                            />
                        )}
                    </div>

                    {feedbackMessage && (
                        <div
                            style={{
                                fontSize: "1.1rem",
                                fontWeight: 600,
                                color:
                                    currentBalloon.status === "exploded" ? "#EF4444" : "#22C55E",
                                minHeight: "2rem",
                            }}
                        >
                            {feedbackMessage}
                        </div>
                    )}

                    <div style={{ display: "flex", gap: "1rem" }}>
                        <button
                            onClick={handlePump}
                            disabled={
                                gamePhase !== "playing" || currentBalloon.status !== "active"
                            }
                            style={{
                                padding: "12px 32px",
                                fontSize: "1rem",
                                fontWeight: 600,
                                color: "#fff",
                                background:
                                    gamePhase !== "playing" || currentBalloon.status !== "active"
                                        ? "#374151"
                                        : "linear-gradient(135deg, #F97316, #EF4444)",
                                border: "none",
                                borderRadius: "10px",
                                cursor:
                                    gamePhase !== "playing" || currentBalloon.status !== "active"
                                        ? "not-allowed"
                                        : "pointer",
                                transition: "transform 0.1s",
                                boxShadow: "0 3px 10px rgba(249,115,22,0.3)",
                            }}
                        >
                            {t.pumpButton}
                        </button>
                        <button
                            onClick={handleCollect}
                            disabled={
                                gamePhase !== "playing" ||
                                currentBalloon.status !== "active" ||
                                currentBalloon.pumps === 0
                            }
                            style={{
                                padding: "12px 32px",
                                fontSize: "1rem",
                                fontWeight: 600,
                                color: "#fff",
                                background:
                                    gamePhase !== "playing" ||
                                        currentBalloon.status !== "active" ||
                                        currentBalloon.pumps === 0
                                        ? "#374151"
                                        : "linear-gradient(135deg, #22C55E, #14B8A6)",
                                border: "none",
                                borderRadius: "10px",
                                cursor:
                                    gamePhase !== "playing" ||
                                        currentBalloon.status !== "active" ||
                                        currentBalloon.pumps === 0
                                        ? "not-allowed"
                                        : "pointer",
                                transition: "transform 0.1s",
                                boxShadow: "0 3px 10px rgba(34,197,94,0.3)",
                            }}
                        >
                            {t.collectButton}
                        </button>
                    </div>

                    <div
                        style={{
                            display: "flex",
                            flexWrap: "wrap",
                            gap: "6px",
                            marginTop: "0.5rem",
                            maxWidth: "400px",
                            justifyContent: "center",
                        }}
                    >
                        {completedBalloons.map((b) => (
                            <div
                                key={b.id}
                                title={`Balon ${b.id}: ${b.pumps} pompa — ${b.status === "collected" ? "toplandı" : "patladı"}`}
                                style={{
                                    width: 24,
                                    height: 24,
                                    borderRadius: "50%",
                                    background:
                                        b.status === "collected"
                                            ? "#22C55E"
                                            : "#EF4444",
                                    opacity: 0.7,
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    fontSize: "0.6rem",
                                    color: "#fff",
                                    fontWeight: 700,
                                }}
                            >
                                {b.pumps}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ── Game Over → Submit ────────────────────────────────────────────── */}
            {gamePhase === "finished" && (
                <div className="flex flex-col items-center py-16 gap-6">
                    <div className="text-5xl">🏁</div>
                    <h2
                        style={{ fontSize: "1.5rem", fontWeight: 700, color: "#fff" }}
                    >
                        {t.finishedTitle}
                    </h2>
                    <p style={{ color: "#9CA3AF" }}>
                        {t.totalEarnings}:{" "}
                        <span style={{ color: "#22C55E", fontWeight: 700 }}>
                            ${totalScore.toFixed(2)}
                        </span>{" "}
                        / {totalBalloons} {t.balloonsWord}
                    </p>

                    <div
                        style={{
                            display: "flex",
                            gap: "6px",
                            flexWrap: "wrap",
                            justifyContent: "center",
                        }}
                    >
                        {completedBalloons.map((b) => (
                            <div
                                key={b.id}
                                style={{
                                    width: 32,
                                    height: 32,
                                    borderRadius: "50%",
                                    background:
                                        b.status === "collected" ? "#22C55E" : "#EF4444",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    fontSize: "0.7rem",
                                    color: "#fff",
                                    fontWeight: 700,
                                }}
                            >
                                {b.pumps}
                            </div>
                        ))}
                    </div>

                    <button
                        onClick={handleSubmit}
                        disabled={isSubmitting}
                        style={{
                            marginTop: "1rem",
                            padding: "14px 48px",
                            fontSize: "1.05rem",
                            fontWeight: 600,
                            color: "#fff",
                            background: isSubmitting
                                ? "#374151"
                                : "linear-gradient(135deg, #6366F1, #8B5CF6)",
                            border: "none",
                            borderRadius: "12px",
                            cursor: isSubmitting ? "wait" : "pointer",
                            boxShadow: "0 4px 15px rgba(99,102,241,0.4)",
                        }}
                    >
                        {isSubmitting ? t.analyzing : t.seeResults}
                    </button>

                    {feedbackMessage && (
                        <p style={{ color: "#EF4444", fontSize: "0.9rem" }}>
                            {feedbackMessage}
                        </p>
                    )}
                </div>
            )}

            {/* ── Debrief (results screen) ─────────────────────────────────────── */}
            {gamePhase === "results" && results && (
                <div className="flex flex-col items-center gap-6">
                    <Debrief results={results} language={config.language} />
                    <button
                        onClick={startGame}
                        style={{
                            marginBottom: "2rem",
                            padding: "12px 36px",
                            fontSize: "1rem",
                            fontWeight: 600,
                            color: "#fff",
                            background: "rgba(255,255,255,0.08)",
                            border: "1px solid rgba(255,255,255,0.15)",
                            borderRadius: "10px",
                            cursor: "pointer",
                        }}
                    >
                        Tekrar Oyna
                    </button>
                </div>
            )}
        </div>
    );
}
