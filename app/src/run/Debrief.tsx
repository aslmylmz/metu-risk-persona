import type { Language } from "../lib/config";
import { taskStrings } from "../lib/i18n";

/** One color's summary row in the debrief (mirrors the engine's color_metrics). */
interface ColorMetrics {
  color: string;
  average_pumps: number;
  explosion_rate: number;
  total_balloons: number;
  risk_profile: string;
}

/** The scored result the sidecar returns (mirrors scoring AssessmentResponse). */
export interface AssessmentResult {
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
  profile_traits: Record<string, { level: string; percentile: number; z_score: number }>;
}

// Results-screen metric labels remain Turkish for now (the bilingual metric pass is
// a deferred follow-up); only the debrief title follows the study language so far.
const COLOR_TR: Record<string, string> = { purple: "Mor", teal: "Camgöbeği", orange: "Turuncu" };
const RISK_TR: Record<string, string> = { Low: "Düşük", Medium: "Orta", High: "Yüksek" };

interface DebriefProps {
  results: AssessmentResult;
  language: Language;
}

/** The participant debrief — the engagement-only results screen, rendered after a
 * finished session. Pure display of one scored `AssessmentResult`; carries no
 * gameplay logic. Split out of BartGame so the task stays a thin shell. */
export function Debrief({ results, language }: DebriefProps) {
  const strings = taskStrings(language);
  return (
    <div className="flex flex-col items-center py-10 gap-6">
      <div className="text-5xl">🎯</div>
      <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "#fff" }}>{strings.debriefTitle}</h2>

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
            Uyum Stratejisi Puanı
          </div>
          {results.raw_metrics.adaptive_strategy_score.toFixed(0)}/100
        </div>
      )}

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
            Balon Rengine Göre Performans
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
                        cm.color === "purple" ? "#A855F7" : cm.color === "teal" ? "#14B8A6" : "#F97316",
                    }}
                  />
                  <span style={{ color: "#fff", fontWeight: 600 }}>
                    {COLOR_TR[cm.color] ?? cm.color} ({RISK_TR[cm.risk_profile] ?? cm.risk_profile} risk)
                  </span>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "#fff" }}>
                    {cm.average_pumps.toFixed(1)} pompa
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "#6B7280" }}>
                    %{(cm.explosion_rate * 100).toFixed(0)} patladı
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

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
          Öğrenme ve Uyum
        </h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
          {[
            {
              label: "Öğrenme Hızı",
              value: results.raw_metrics.learning_rate?.toFixed(2) || "0.00",
              subtitle: "Davranışsal uyum",
            },
            {
              label: "Renk Ayrımı",
              value: results.raw_metrics.color_discrimination_index?.toFixed(2) || "0.00",
              subtitle: "Örüntü tanıma",
            },
            {
              label: "Risk Ayarlaması",
              value: results.raw_metrics.risk_adjustment_score?.toFixed(0) || "0",
              subtitle: "Strateji kalibrasyonu",
            },
            {
              label: "Tepki Tutarlılığı",
              value: results.raw_metrics.response_consistency?.toFixed(2) || "0.00",
              subtitle: "Davranışsal istikrar",
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
              <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#fff", marginBottom: "2px" }}>
                {metric.value}
              </div>
              <div style={{ fontSize: "0.65rem", color: "#6B7280" }}>{metric.subtitle}</div>
            </div>
          ))}
        </div>
      </div>

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
          Davranışsal Göstergeler
        </h3>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px" }}>
          {[
            {
              label: "Dürtüsellik",
              value: `%${((results.raw_metrics.impulsivity_index || 0) * 100).toFixed(0)}`,
              subtitle: "Yüksek riskli patlamalar",
            },
            {
              label: "Sabır",
              value: results.raw_metrics.patience_index?.toFixed(1) || "0.0",
              subtitle: "Düşük riskli istismar",
            },
            {
              label: "Ort. Pompa",
              value: results.raw_metrics.average_pumps_adjusted.toFixed(1),
              subtitle: "Genel risk iştahı",
            },
            {
              label: "Patlama Oranı",
              value: `%${(results.raw_metrics.explosion_rate * 100).toFixed(0)}`,
              subtitle: "Başarısızlık oranı",
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
              <div style={{ fontSize: "1.3rem", fontWeight: 700, color: "#fff", marginBottom: "2px" }}>
                {metric.value}
              </div>
              <div style={{ fontSize: "0.65rem", color: "#6B7280" }}>{metric.subtitle}</div>
            </div>
          ))}
        </div>
      </div>

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
            Popülasyon Karşılaştırması
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
                <span style={{ color: "#fff", fontWeight: 600, fontSize: "0.95rem" }}>
                  {score.percentile.toFixed(0)}.
                </span>
                <span style={{ color: "#6B7280", fontSize: "0.75rem", marginLeft: "6px" }}>
                  (z={score.z_score.toFixed(2)})
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
