// Recreated from the reference screenshot (no direct file access to the
// original asset) — colors and shape approximate the source logo. Drop the
// real file at frontend/public/logo.png and swap the JSX below for
// <img src="/logo.png" alt="WGDeepInsight" /> for a pixel-exact match.

export default function Logo() {
  return (
    <div className="fqc-logo">
      <svg width="34" height="34" viewBox="0 0 40 40" aria-hidden="true">
        <path
          d="M20 2 L36 8 V19 C36 28 29 35 20 38 C11 35 4 28 4 19 V8 Z"
          fill="none"
          stroke="#1d5a96"
          strokeWidth="2.4"
        />
        <circle cx="20" cy="19" r="7" fill="#c9ccd1" />
        <circle cx="20" cy="19" r="5" fill="#3a3a3a" />
        <circle cx="18" cy="17" r="1.6" fill="#fff" />
        <path d="M9 14 Q13 10 16 13" fill="none" stroke="#4a9fd8" strokeWidth="1.6" strokeLinecap="round" />
        <path d="M31 14 Q27 10 24 13" fill="none" stroke="#4a9fd8" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      <div className="fqc-logo-text">
        <div className="fqc-logo-wordmark">
          <span style={{ color: "var(--brand-orange)" }}>WG</span>
          <span style={{ color: "var(--brand-blue)" }}>Deep</span>
          <span style={{ color: "var(--brand-orange)" }}>Insight</span>
        </div>
        <div className="fqc-logo-tagline">Detect. Predict. Protect.</div>
      </div>
    </div>
  );
}
