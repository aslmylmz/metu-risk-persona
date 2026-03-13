"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// ── Types ───────────────────────────────────────────────────────────────────

interface GameEvent {
    timestamp: number;
    type: "pump" | "collect" | "explode";
    payload: Record<string, unknown>;
}

interface BalloonState {
    id: number;
    pumps: number;
    status: "active" | "collected" | "exploded";
}

interface ColorMetrics {
    color: string;
    average_pumps: number;
    explosion_rate: number;
    total_balloons: number;
    risk_profile: string;
}

interface AssessmentResult {
    session_id: string;
    game_type: string;
    raw_metrics: {
        average_pumps_adjusted: number;
        explosion_rate: number;
        mean_latency_between_pumps: number;
        total_balloons: number;
        total_pumps: number;
        total_explosions: number;
        total_collections: number;
        // New multi-risk metrics
        color_metrics: ColorMetrics[];
        learning_rate: number;
        risk_adjustment_score: number;
        color_discrimination_index: number;
        impulsivity_index: number;
        patience_index: number;
        response_consistency: number;
        adaptive_strategy_score: number;
    };
    normalized_scores: Array<{
        metric_name: string;
        raw_value: number;
        z_score: number;
        percentile: number;
    }>;
    profile_traits: Record<
        string,
        { level: string; percentile: number; z_score: number }
    >;
}

// ── Config ──────────────────────────────────────────────────────────────────

// ── Config ──────────────────────────────────────────────────────────────────

const TOTAL_BALLOONS = 30; // 10 Red, 10 Yellow, 10 Blue

interface RiskProfile {
    color: string;
    maxPumps: number;
    riskLevel: "Low" | "Medium" | "High";
}

const RISK_PROFILES: Record<string, RiskProfile> = {
    ORANGE: { color: "#F97316", maxPumps: 8, riskLevel: "High" },    // High Risk (Pops early)
    TEAL: { color: "#14B8A6", maxPumps: 32, riskLevel: "Medium" },   // Standard Risk
    PURPLE: { color: "#A855F7", maxPumps: 128, riskLevel: "Low" },   // Low Risk (Lasts long)
};

interface BalloonConfig {
    id: number;
    color: string;
    maxPumps: number;
}

/**
 * Generates a shuffled session with 10 of each balloon type.
 * Each balloon has a linear explosion probability: P(pop at pump k) = k / maxPumps
 */
function generateSessionConfig(): BalloonConfig[] {
    const configs: BalloonConfig[] = [];

    // Generate 10 of each type
    (["ORANGE", "TEAL", "PURPLE"] as const).forEach((type) => {
        const profile = RISK_PROFILES[type];
        for (let i = 0; i < 10; i++) {
            configs.push({
                id: 0, // Assigned later
                color: profile.color,
                maxPumps: profile.maxPumps,
            });
        }
    });

    // Fisher-Yates Shuffle
    for (let i = configs.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [configs[i], configs[j]] = [configs[j], configs[i]];
    }

    // Assign IDs matches visual order
    return configs.map((c, idx) => ({ ...c, id: idx + 1 }));
}

// ── Component ───────────────────────────────────────────────────────────────

interface BartGameProps {
    candidateId: string;
    /** Called with backend response after successful submission */
    onComplete?: (data: AssessmentResult) => void;
}

export default function BartGame({ candidateId, onComplete }: BartGameProps) {
    // ── Raw event log (useRef — no re-renders) ────────────────────────────────
    // CRITICAL: Uses performance.now(), NEVER Date.now()
    const eventLogRef = useRef<GameEvent[]>([]);
    const sessionIdRef = useRef(crypto.randomUUID());
    // Stores the full configuration (color, limit, id) for the session
    const sessionConfigRef = useRef<BalloonConfig[]>([]);
    const containerRef = useRef<HTMLDivElement>(null);

    // ── UI state ──────────────────────────────────────────────────────────────
    const [currentBalloon, setCurrentBalloon] = useState<BalloonState>({
        id: 1,
        pumps: 0,
        status: "active",
    });
    const [completedBalloons, setCompletedBalloons] = useState<BalloonState[]>(
        []
    );
    const [totalScore, setTotalScore] = useState(0);
    const [gamePhase, setGamePhase] = useState<
        "idle" | "playing" | "feedback" | "finished" | "results"
    >("idle");
    const [feedbackMessage, setFeedbackMessage] = useState("");
    const [results, setResults] = useState<AssessmentResult | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    const balloonCount = completedBalloons.length + 1;

    // ── Record event (performance.now() — NEVER Date.now()) ───────────────────
    const recordEvent = useCallback(
        (type: GameEvent["type"], extra: Record<string, unknown> = {}) => {
            // Get current balloon config for color
            const config = sessionConfigRef.current[currentBalloon.id - 1];

            if (!config) {
                console.error(`No config found for balloon ${currentBalloon.id}`, {
                    configLength: sessionConfigRef.current.length,
                    balloonId: currentBalloon.id
                });
            }

            const colorName = config?.color === "#F97316" ? "orange" :
                              config?.color === "#14B8A6" ? "teal" :
                              config?.color === "#A855F7" ? "purple" : "teal";

            console.log(`Event ${type} - Balloon ${currentBalloon.id}: color=${colorName}, hexColor=${config?.color}`);

            eventLogRef.current.push({
                timestamp: performance.now(), // Monotonic high-resolution timer
                type,
                payload: {
                    balloon_id: currentBalloon.id,
                    color: colorName,
                    ...extra
                },
            });
        },
        [currentBalloon.id]
    );

    // ── Game actions ──────────────────────────────────────────────────────────

    const startGame = useCallback(() => {
        eventLogRef.current = [];
        sessionIdRef.current = crypto.randomUUID();
        // Generate new Multi-Risk session
        sessionConfigRef.current = generateSessionConfig();

        setCurrentBalloon({ id: 1, pumps: 0, status: "active" });
        setCompletedBalloons([]);
        setTotalScore(0);
        setResults(null);
        setGamePhase("playing");
    }, []);

    const handlePump = useCallback(() => {
        if (gamePhase !== "playing" || currentBalloon.status !== "active") return;

        const newPumps = currentBalloon.pumps + 1;
        recordEvent("pump");

        // Get config for current balloon (0-indexed)
        const config = sessionConfigRef.current[currentBalloon.id - 1];
        const maxPumps = config ? config.maxPumps : 32;

        // BART probability model: P(pop) = currentPumps / maxPumps (linear increase)
        // Balloon ALWAYS pops if we reach maxPumps
        const explosionProbability = newPumps / maxPumps;
        const explode = newPumps >= maxPumps || Math.random() < explosionProbability;

        if (explode) {
            recordEvent("explode", { pump_count: newPumps });
            setCurrentBalloon((b) => ({ ...b, pumps: newPumps, status: "exploded" }));
            setFeedbackMessage("POPPED! $0.00 for this balloon.");
            setGamePhase("feedback");

            setTimeout(() => {
                const exploded: BalloonState = {
                    id: currentBalloon.id,
                    pumps: newPumps,
                    status: "exploded",
                };
                setCompletedBalloons((prev) => {
                    const updated = [...prev, exploded];
                    if (updated.length >= TOTAL_BALLOONS) {
                        setGamePhase("finished");
                    } else {
                        setCurrentBalloon({
                            id: currentBalloon.id + 1,
                            pumps: 0,
                            status: "active",
                        });
                        setGamePhase("playing");
                    }
                    return updated;
                });
                setFeedbackMessage("");
            }, 1200);
        } else {
            setCurrentBalloon((b) => ({ ...b, pumps: newPumps }));
        }
    }, [gamePhase, currentBalloon, recordEvent]);

    const handleCollect = useCallback(() => {
        if (
            gamePhase !== "playing" ||
            currentBalloon.status !== "active" ||
            currentBalloon.pumps === 0
        )
            return;

        recordEvent("collect");

        const money = currentBalloon.pumps * 0.25;
        setTotalScore((s) => s + money);
        setCurrentBalloon((b) => ({ ...b, status: "collected" }));
        setFeedbackMessage(`Collected $${money.toFixed(2)}!`);
        setGamePhase("feedback");

        setTimeout(() => {
            const collected: BalloonState = {
                id: currentBalloon.id,
                pumps: currentBalloon.pumps,
                status: "collected",
            };
            setCompletedBalloons((prev) => {
                const updated = [...prev, collected];
                if (updated.length >= TOTAL_BALLOONS) {
                    setGamePhase("finished");
                } else {
                    setCurrentBalloon({
                        id: currentBalloon.id + 1,
                        pumps: 0,
                        status: "active",
                    });
                    setGamePhase("playing");
                }
                return updated;
            });
            setFeedbackMessage("");
        }, 1000);
    }, [gamePhase, currentBalloon, recordEvent]);

    // ── Submit to backend ─────────────────────────────────────────────────────

    const handleSubmit = useCallback(async () => {
        setIsSubmitting(true);

        const payload = {
            session_id: sessionIdRef.current,
            game_type: "BART_RISK",
            candidate_id: candidateId,
            events: eventLogRef.current,
        };

        // DEBUG: Log submission data
        const colorDist: Record<string, number> = {};
        payload.events.forEach(e => {
            const color = String(e.payload.color || "MISSING");
            colorDist[color] = (colorDist[color] || 0) + 1;
        });
        console.log("Submitting BART assessment:", {
            totalEvents: payload.events.length,
            sampleEvents: payload.events.slice(0, 5),
            colorDistribution: colorDist
        });

        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            const response = await fetch(
                `${apiUrl}/assessments/bart`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Scoring failed");
            }

            const data: AssessmentResult = await response.json();
            setResults(data);
            setGamePhase("results");

            // Notify orchestrator if in managed mode
            if (onComplete) {
                onComplete(data);
            }
        } catch (err) {
            console.error("Submission error:", err);
            setFeedbackMessage(
                err instanceof Error ? err.message : "Failed to submit"
            );
        } finally {
            setIsSubmitting(false);
        }
    }, [candidateId]);

    // ── Keyboard shortcuts ────────────────────────────────────────────────────

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent) => {
            if (e.code === "Space") {
                e.preventDefault();
                handlePump();
            } else if (e.code === "Enter") {
                e.preventDefault();
                handleCollect();
            }
        },
        [handlePump, handleCollect]
    );

    // ── Auto-focus for keyboard controls ───────────────────────────────────────
    useEffect(() => {
        if (gamePhase === "playing" && containerRef.current) {
            containerRef.current.focus();
        }
    }, [gamePhase]);

    // ── Debug logging (client-side only) ──────────────────────────────────────
    useEffect(() => {
        if (gamePhase === "playing" && sessionConfigRef.current.length > 0) {
            // Log balloon sequence generation
            console.log("Generated balloon sequence:", sessionConfigRef.current.map((c) => ({
                id: c.id,
                colorHex: c.color,
                colorName: c.color === "#F97316" ? "ORANGE" : c.color === "#14B8A6" ? "TEAL" : c.color === "#A855F7" ? "PURPLE" : "UNKNOWN",
                maxPumps: c.maxPumps
            })));

            const colorCounts = sessionConfigRef.current.reduce((acc, c) => {
                const name = c.color === "#F97316" ? "ORANGE" : c.color === "#14B8A6" ? "TEAL" : c.color === "#A855F7" ? "PURPLE" : "UNKNOWN";
                acc[name] = (acc[name] || 0) + 1;
                return acc;
            }, {} as Record<string, number>);
            console.log("Color distribution:", colorCounts);
        }
    }, [gamePhase]);

    // ── Balloon scale ─────────────────────────────────────────────────────────

    // Dynamic color based on current balloon ID
    const currentConfig = sessionConfigRef.current[currentBalloon.id - 1];
    const balloonColor = currentConfig ? currentConfig.color : "#9CA3AF";
    const balloonScale = 1 + currentBalloon.pumps * 0.08;
    const balloonSize = 100 * balloonScale;

    // ── Render ────────────────────────────────────────────────────────────────

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
                        Balloon Analogue Risk Task
                    </h2>
                    <p
                        style={{
                            color: "#9CA3AF",
                            textAlign: "center",
                            maxWidth: "400px",
                            lineHeight: 1.6,
                        }}
                    >
                        Pump the balloon to earn money ($0.25 per pump). Each pump increases the risk of
                        popping. Collect your money before it pops!
                    </p>
                    <p
                        style={{ color: "#6B7280", fontSize: "0.85rem" }}
                    >
                        {TOTAL_BALLOONS} balloons · Space to pump · Enter to collect
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
                        Start Assessment
                    </button>
                </div>
            )}

            {/* ── Playing / Feedback ───────────────────────────────────────────── */}
            {(gamePhase === "playing" || gamePhase === "feedback") && (
                <div className="flex flex-col items-center py-8 gap-6">
                    {/* Status Bar */}
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
                            Balloon{" "}
                            <span style={{ color: "#fff", fontWeight: 700 }}>
                                {Math.min(balloonCount, TOTAL_BALLOONS)}
                            </span>
                            /{TOTAL_BALLOONS}
                        </span>
                        <span style={{ color: "#9CA3AF", fontSize: "0.9rem" }}>
                            Total{" "}
                            <span style={{ color: "#22C55E", fontWeight: 700 }}>
                                ${totalScore.toFixed(2)}
                            </span>
                        </span>
                    </div>

                    {/* Balloon */}
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
                                {/* Highlight */}
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
                                {/* Money amount */}
                                <span
                                    style={{
                                        fontSize: `${Math.max(1.2, 1.8 * balloonScale * 0.4)}rem`,
                                        fontWeight: 800,
                                        color: "rgba(255,255,255,0.9)",
                                        textShadow: "0 2px 4px rgba(0,0,0,0.3)",
                                    }}
                                >
                                    ${(currentBalloon.pumps * 0.25).toFixed(2)}
                                </span>
                            </div>
                        )}

                        {/* String */}
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

                    {/* Feedback */}
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

                    {/* Action buttons */}
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
                            🎈 Pump (Space)
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
                            💰 Collect (Enter)
                        </button>
                    </div>

                    {/* Mini balloon history */}
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
                                title={`Balloon ${b.id}: ${b.pumps} pumps — ${b.status}`}
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
                        Assessment Complete
                    </h2>
                    <p style={{ color: "#9CA3AF" }}>
                        Total Earned:{" "}
                        <span style={{ color: "#22C55E", fontWeight: 700 }}>
                            ${totalScore.toFixed(2)}
                        </span>{" "}
                        across {TOTAL_BALLOONS} balloons
                    </p>

                    {/* Balloon history summary */}
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
                        {isSubmitting ? "Analyzing..." : onComplete ? "Next" : "View My Results"}
                    </button>

                    {feedbackMessage && (
                        <p style={{ color: "#EF4444", fontSize: "0.9rem" }}>
                            {feedbackMessage}
                        </p>
                    )}
                </div>
            )}

            {/* ── Results Modal ────────────────────────────────────────────────── */}
            {gamePhase === "results" && results && (
                <div className="flex flex-col items-center py-10 gap-6">
                    <div className="text-5xl">🎯</div>
                    <h2
                        style={{ fontSize: "1.5rem", fontWeight: 700, color: "#fff" }}
                    >
                        Your Cognitive Profile
                    </h2>

                    {/* Adaptive Strategy Score - Hero Metric */}
                    {results.raw_metrics.adaptive_strategy_score !== undefined && (
                        <div
                            style={{
                                fontSize: "1.6rem",
                                fontWeight: 800,
                                padding: "16px 32px",
                                borderRadius: "12px",
                                background: "linear-gradient(135deg, #6366F1, #8B5CF6)",
                                border: "1px solid rgba(255,255,255,0.2)",
                                textAlign: "center",
                            }}
                        >
                            <div style={{ fontSize: "0.75rem", opacity: 0.9, marginBottom: "4px" }}>
                                Adaptive Strategy Score
                            </div>
                            {results.raw_metrics.adaptive_strategy_score.toFixed(0)}/100
                        </div>
                    )}

                    {/* Color-Based Performance */}
                    {results.raw_metrics.color_metrics && results.raw_metrics.color_metrics.length > 0 && (
                        <div style={{ width: "100%", maxWidth: "480px" }}>
                            <h3
                                style={{
                                    fontSize: "0.85rem",
                                    color: "#9CA3AF",
                                    marginBottom: "12px",
                                    textTransform: "uppercase",
                                    letterSpacing: "0.05em",
                                    textAlign: "center",
                                }}
                            >
                                Performance by Balloon Color
                            </h3>
                            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                                {results.raw_metrics.color_metrics.map((cm) => (
                                    <div
                                        key={cm.color}
                                        style={{
                                            padding: "12px 16px",
                                            borderRadius: "10px",
                                            background: "rgba(255,255,255,0.03)",
                                            border: "1px solid rgba(255,255,255,0.08)",
                                            display: "flex",
                                            justifyContent: "space-between",
                                            alignItems: "center",
                                        }}
                                    >
                                        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                                            <div
                                                style={{
                                                    width: "12px",
                                                    height: "12px",
                                                    borderRadius: "50%",
                                                    background:
                                                        cm.color === "purple" ? "#A855F7" :
                                                        cm.color === "teal" ? "#14B8A6" :
                                                        "#F97316",
                                                }}
                                            />
                                            <span style={{ color: "#fff", fontWeight: 600, textTransform: "capitalize" }}>
                                                {cm.color} ({cm.risk_profile} risk)
                                            </span>
                                        </div>
                                        <div style={{ textAlign: "right" }}>
                                            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#fff" }}>
                                                {cm.average_pumps.toFixed(1)} pumps
                                            </div>
                                            <div style={{ fontSize: "0.75rem", color: "#6B7280" }}>
                                                {(cm.explosion_rate * 100).toFixed(0)}% exploded
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Learning & Adaptation Metrics */}
                    <div style={{ width: "100%", maxWidth: "480px" }}>
                        <h3
                            style={{
                                fontSize: "0.85rem",
                                color: "#9CA3AF",
                                marginBottom: "12px",
                                textTransform: "uppercase",
                                letterSpacing: "0.05em",
                                textAlign: "center",
                            }}
                        >
                            Learning & Adaptation
                        </h3>
                        <div
                            style={{
                                display: "grid",
                                gridTemplateColumns: "1fr 1fr",
                                gap: "10px",
                            }}
                        >
                            {[
                                {
                                    label: "Learning Rate",
                                    value: results.raw_metrics.learning_rate?.toFixed(2) || "0.00",
                                    subtitle: "Behavioral adaptation",
                                },
                                {
                                    label: "Color Discrimination",
                                    value: results.raw_metrics.color_discrimination_index?.toFixed(2) || "0.00",
                                    subtitle: "Pattern recognition",
                                },
                                {
                                    label: "Risk Adjustment",
                                    value: results.raw_metrics.risk_adjustment_score?.toFixed(0) || "0",
                                    subtitle: "Strategy calibration",
                                },
                                {
                                    label: "Response Consistency",
                                    value: results.raw_metrics.response_consistency?.toFixed(2) || "0.00",
                                    subtitle: "Behavioral stability",
                                },
                            ].map((metric) => (
                                <div
                                    key={metric.label}
                                    style={{
                                        padding: "12px",
                                        borderRadius: "10px",
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.08)",
                                        textAlign: "center",
                                    }}
                                >
                                    <div
                                        style={{
                                            fontSize: "0.7rem",
                                            color: "#6B7280",
                                            textTransform: "uppercase",
                                            letterSpacing: "0.05em",
                                            marginBottom: "4px",
                                        }}
                                    >
                                        {metric.label}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: "1.3rem",
                                            fontWeight: 700,
                                            color: "#fff",
                                            marginBottom: "2px",
                                        }}
                                    >
                                        {metric.value}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: "0.65rem",
                                            color: "#6B7280",
                                        }}
                                    >
                                        {metric.subtitle}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Behavioral Indices */}
                    <div style={{ width: "100%", maxWidth: "480px" }}>
                        <h3
                            style={{
                                fontSize: "0.85rem",
                                color: "#9CA3AF",
                                marginBottom: "12px",
                                textTransform: "uppercase",
                                letterSpacing: "0.05em",
                                textAlign: "center",
                            }}
                        >
                            Behavioral Indices
                        </h3>
                        <div
                            style={{
                                display: "grid",
                                gridTemplateColumns: "1fr 1fr",
                                gap: "10px",
                            }}
                        >
                            {[
                                {
                                    label: "Impulsivity",
                                    value: `${((results.raw_metrics.impulsivity_index || 0) * 100).toFixed(0)}%`,
                                    subtitle: "High-risk explosions",
                                },
                                {
                                    label: "Patience",
                                    value: results.raw_metrics.patience_index?.toFixed(1) || "0.0",
                                    subtitle: "Low-risk exploitation",
                                },
                                {
                                    label: "Avg Pumps",
                                    value: results.raw_metrics.average_pumps_adjusted.toFixed(1),
                                    subtitle: "Overall risk appetite",
                                },
                                {
                                    label: "Explosion Rate",
                                    value: `${(results.raw_metrics.explosion_rate * 100).toFixed(0)}%`,
                                    subtitle: "Failure rate",
                                },
                            ].map((metric) => (
                                <div
                                    key={metric.label}
                                    style={{
                                        padding: "12px",
                                        borderRadius: "10px",
                                        background: "rgba(255,255,255,0.03)",
                                        border: "1px solid rgba(255,255,255,0.08)",
                                        textAlign: "center",
                                    }}
                                >
                                    <div
                                        style={{
                                            fontSize: "0.7rem",
                                            color: "#6B7280",
                                            textTransform: "uppercase",
                                            letterSpacing: "0.05em",
                                            marginBottom: "4px",
                                        }}
                                    >
                                        {metric.label}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: "1.3rem",
                                            fontWeight: 700,
                                            color: "#fff",
                                            marginBottom: "2px",
                                        }}
                                    >
                                        {metric.value}
                                    </div>
                                    <div
                                        style={{
                                            fontSize: "0.65rem",
                                            color: "#6B7280",
                                        }}
                                    >
                                        {metric.subtitle}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Normalized scores */}
                    {results.normalized_scores.length > 0 && (
                        <div style={{ width: "100%", maxWidth: "420px" }}>
                            <h3
                                style={{
                                    fontSize: "0.9rem",
                                    color: "#9CA3AF",
                                    marginBottom: "8px",
                                    textTransform: "uppercase",
                                    letterSpacing: "0.05em",
                                }}
                            >
                                Population Comparison
                            </h3>
                            {results.normalized_scores.map((score) => (
                                <div
                                    key={score.metric_name}
                                    style={{
                                        display: "flex",
                                        justifyContent: "space-between",
                                        alignItems: "center",
                                        padding: "10px 14px",
                                        borderBottom: "1px solid rgba(255,255,255,0.05)",
                                    }}
                                >
                                    <span style={{ color: "#9CA3AF", fontSize: "0.85rem" }}>
                                        {score.metric_name.replaceAll("_", " ")}
                                    </span>
                                    <div style={{ textAlign: "right" }}>
                                        <span
                                            style={{
                                                color: "#fff",
                                                fontWeight: 600,
                                                fontSize: "0.95rem",
                                            }}
                                        >
                                            {score.percentile.toFixed(0)}th
                                        </span>
                                        <span
                                            style={{
                                                color: "#6B7280",
                                                fontSize: "0.75rem",
                                                marginLeft: "6px",
                                            }}
                                        >
                                            (z={score.z_score.toFixed(2)})
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Play again */}
                    <button
                        onClick={startGame}
                        style={{
                            marginTop: "1rem",
                            padding: "12px 36px",
                            fontSize: "1rem",
                            fontWeight: 600,
                            color: "#fff",
                            background: "rgba(255,255,255,0.08)",
                            border: "1px solid rgba(255,255,255,0.15)",
                            borderRadius: "10px",
                            cursor: "pointer",
                            transition: "background 0.15s",
                        }}
                        onMouseEnter={(e) => {
                            e.currentTarget.style.background = "rgba(255,255,255,0.12)";
                        }}
                        onMouseLeave={(e) => {
                            e.currentTarget.style.background = "rgba(255,255,255,0.08)";
                        }}
                    >
                        Play Again
                    </button>
                </div>
            )}
        </div>
    );
}
