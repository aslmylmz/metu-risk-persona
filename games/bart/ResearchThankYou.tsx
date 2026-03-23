"use client";

import { useState } from "react";

// ── Research Thank-You Screen ────────────────────────────────────────────────
// Shown as the final screen in BART-only / research mode.
// Displays the participant's BART risk profile archetype above the thank-you.

interface BARTResults {
    session_id: string;
    game_type: string;
    raw_metrics: {
        behavioral_profile?: {
            risk_style?: string;
            description?: string;
            dominant_traits?: string[];
        };
        [key: string]: unknown;
    };
    [key: string]: unknown;
}

interface ResearchThankYouProps {
    candidateId: string;
    bartResults?: BARTResults | null;
    onReset?: () => void;
}

// ── Turkish translations for dominant traits ─────────────────────────────────

const TRAIT_TR: Record<string, string> = {
    "Highly Consistent": "Yüksek Tutarlılık",
    "Erratic (within-balloon)": "Sezgisel Karar Verici",
    "Strategically Variable": "Stratejik Esneklik",
    "Adaptive Learner": "Hızlı Öğrenici",
    "Risk-Averse Learner": "Dikkatli Gözlemci",
    "Impulsive on High-Risk": "Hızlı Tepki Veren",
    "Patient Optimizer": "Sabırlı Optimizatör",
    "Over-Pumper on Safe Balloons": "Cesur Keşifçi",
    "Flat Strategy": "Kararlı Strateji",
    "High Explosion Penalty": "Yüksek Risk İştahı",
};

// ── Turkish translations for risk archetypes ─────────────────────────────────

const RISK_STYLE_TR: Record<string, { title: string; emoji: string; description: string }> = {
    "Calibrated Risk Optimizer": {
        title: "Kalibre Edilmiş Risk Uzmanı",
        emoji: "🎯",
        description:
            "Risk alma kararlarınızı gerçek tehlike seviyelerine göre hassasiyetle kalibre ettiniz. Güvenli olduğunda baskıyı artırdınız, risk yükseldiğinde geri çekildiniz — koşulları okuyan analitik bir karar verme tarzınız var.",
    },
    "Aggressive Risk Taker": {
        title: "Cesur Risk Alan",
        emoji: "🔥",
        description:
            "Belirsizlik karşısında kararlılıkla hareket ediyorsunuz. Fırsatları değerlendirme konusunda güçlü bir motivasyonunuz var ve baskı altında geri adım atmıyorsunuz — bu cesaret, doğru bağlamda büyük avantaj sağlayan bir özellik.",
    },
    "Conservative Safety-Seeker": {
        title: "İhtiyatlı Karar Verici",
        emoji: "🛡️",
        description:
            "Düşünceli ve kontrollü bir yaklaşımınız var. Kararlarınızda güvenliği ön planda tutuyorsunuz ve olası riskleri dikkatlice değerlendiriyorsunuz. Bu disiplinli tarz, özellikle hassas kararlar gerektiren ortamlarda güçlü bir yetkinlik.",
    },
    "Balanced Explorer": {
        title: "Dengeli Kaşif",
        emoji: "⚖️",
        description:
            "Güvenlik ile keşif arasında sağlıklı bir denge kuruyorsunuz. Risklere açık ama ölçülüsünüz; tutarlı ve güvenilir kararlar alıyorsunuz. Bu esneklik, farklı koşullara uyum sağlama kapasitenizi gösteriyor.",
    },
    "Undifferentiated Risk Approach": {
        title: "Tutarlı Strateji Uygulayıcı",
        emoji: "🔄",
        description:
            "Tüm koşullarda istikrarlı ve tutarlı bir karar verme tarzı sergiliyorsunuz. Belirsizlik karşısında sabit bir strateji uygulamak, kararlılık ve özgüven göstergesidir — bu temel üzerine farklı senaryolara uyum sağlama kapasitesi inşa edilebilir.",
    },
};

export default function ResearchThankYou({
    candidateId,
    bartResults,
    onReset,
}: ResearchThankYouProps) {
    const [confirmed, setConfirmed] = useState(false);

    const handleReset = () => {
        if (!confirmed) {
            setConfirmed(true);
            return;
        }
        setConfirmed(false);
        onReset?.();
    };

    const riskStyle = bartResults?.raw_metrics?.behavioral_profile?.risk_style;
    const archetype = riskStyle ? RISK_STYLE_TR[riskStyle] : null;
    const dominantTraits = bartResults?.raw_metrics?.behavioral_profile?.dominant_traits ?? [];

    return (
        <div
            style={{
                maxWidth: 500,
                width: "100%",
                margin: "0 auto",
                padding: "44px 30px",
                borderRadius: 20,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid rgba(255,255,255,0.08)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 22,
                textAlign: "center",
            }}
        >
            {/* ── Risk Profile Archetype ───────────────────────────────── */}
            {archetype && (
                <>
                    <div
                        style={{
                            width: "100%",
                            padding: "24px 24px 20px",
                            borderRadius: 16,
                            background:
                                "linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1))",
                            border: "1px solid rgba(99,102,241,0.25)",
                        }}
                    >
                        <div style={{ fontSize: "2.4rem", marginBottom: 10 }}>
                            {archetype.emoji}
                        </div>
                        <div
                            style={{
                                fontSize: "0.65rem",
                                fontWeight: 600,
                                color: "#818CF8",
                                textTransform: "uppercase",
                                letterSpacing: "0.08em",
                                marginBottom: 6,
                            }}
                        >
                            Risk Profiliniz
                        </div>
                        <h3
                            style={{
                                fontSize: "1.4rem",
                                fontWeight: 700,
                                color: "#fff",
                                margin: "0 0 10px 0",
                            }}
                        >
                            {archetype.title}
                        </h3>
                        <p
                            style={{
                                color: "#A5B4FC",
                                fontSize: "0.85rem",
                                lineHeight: 1.7,
                                margin: 0,
                            }}
                        >
                            {archetype.description}
                        </p>

                        {/* Dominant Traits Badges */}
                        {dominantTraits.length > 0 && (
                            <div
                                style={{
                                    display: "flex",
                                    flexWrap: "wrap",
                                    justifyContent: "center",
                                    gap: 8,
                                    marginTop: 14,
                                }}
                            >
                                {dominantTraits.map((trait) => (
                                    <span
                                        key={trait}
                                        style={{
                                            padding: "4px 12px",
                                            borderRadius: 20,
                                            background: "rgba(129,140,248,0.15)",
                                            border: "1px solid rgba(129,140,248,0.3)",
                                            color: "#C7D2FE",
                                            fontSize: "0.72rem",
                                            fontWeight: 600,
                                        }}
                                    >
                                        {TRAIT_TR[trait] ?? trait}
                                    </span>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Divider */}
                    <div
                        style={{
                            width: "100%",
                            height: 1,
                            background:
                                "linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent)",
                        }}
                    />
                </>
            )}

            {/* ── Thank-You Section ───────────────────────────────────── */}
            <div style={{ fontSize: "2.5rem" }}>✅</div>

            <h2
                style={{
                    fontSize: "1.5rem",
                    fontWeight: 700,
                    color: "#fff",
                    margin: 0,
                }}
            >
                Katılımınız için Teşekkür Ederiz!
            </h2>

            <p
                style={{
                    color: "#9CA3AF",
                    lineHeight: 1.7,
                    fontSize: "0.9rem",
                    margin: 0,
                }}
            >
                Oturumunuz başarıyla kaydedildi. Yanıtlarınız karar verme ve
                risk davranışı araştırmamıza katkı sağlayacaktır.
            </p>

            {/* Session ID box */}
            <div
                style={{
                    width: "100%",
                    padding: "14px 18px",
                    borderRadius: 12,
                    background: "rgba(255,255,255,0.02)",
                    border: "1px solid rgba(255,255,255,0.05)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 6,
                }}
            >
                <div
                    style={{
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        color: "#6B7280",
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                    }}
                >
                    Katılımcı kimliğiniz
                </div>
                <code
                    style={{
                        color: "#A78BFA",
                        background: "rgba(139,92,246,0.1)",
                        border: "1px solid rgba(139,92,246,0.2)",
                        borderRadius: 8,
                        padding: "6px 14px",
                        fontSize: "0.8rem",
                        fontFamily: "monospace",
                        wordBreak: "break-all",
                    }}
                >
                    {candidateId}
                </code>
                <p
                    style={{
                        color: "#6B7280",
                        fontSize: "0.72rem",
                        margin: 0,
                    }}
                >
                    Gerekirse bu kimliği kayıt olarak saklayın.
                </p>
            </div>

            {/* Close prompt */}
            <p
                style={{
                    color: "#6B7280",
                    fontSize: "0.85rem",
                    margin: 0,
                }}
            >
                Bu pencereyi şimdi kapatabilirsiniz.
            </p>

            {/* Researcher: next participant button */}
            {onReset && (
                <button
                    onClick={handleReset}
                    style={{
                        marginTop: 4,
                        padding: "12px 32px",
                        fontSize: "0.95rem",
                        fontWeight: 600,
                        color: confirmed ? "#fff" : "#9CA3AF",
                        background: confirmed
                            ? "linear-gradient(135deg, #10B981, #059669)"
                            : "rgba(255,255,255,0.05)",
                        border: confirmed
                            ? "1px solid rgba(16,185,129,0.4)"
                            : "1px solid rgba(255,255,255,0.12)",
                        borderRadius: 12,
                        cursor: "pointer",
                        transition: "all 0.2s",
                        width: "100%",
                    }}
                >
                    {confirmed
                        ? "✓ Onaylayın — Yeni Katılımcı Başlatılıyor"
                        : "Yeni Katılımcı →"}
                </button>
            )}
        </div>
    );
}
