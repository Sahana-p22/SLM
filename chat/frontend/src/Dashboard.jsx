import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE } from "./api";
import Logo from "./Logo";
import "./Dashboard.css";

const PRESETS = [
  { label: "15m", hours: 0.25 },
  { label: "1h", hours: 1 },
  { label: "6h", hours: 6 },
  { label: "24h", hours: 24 },
  { label: "7d", hours: 24 * 7 },
];

function toLocalInputValue(date) {
  const pad = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function StatCard({ label, value, accent, hint }) {
  return (
    <div className="fqc-stat-card" style={{ "--stat-accent": accent }}>
      <div className="fqc-stat-value">{value}</div>
      <div className="fqc-stat-label">{label}</div>
      {hint && <div className="fqc-stat-hint">{hint}</div>}
    </div>
  );
}

function TypePill({ type }) {
  return <span className={`type-pill type-${type}`}>{type.replace("_", " ")}</span>;
}

export default function Dashboard({ refreshSignal }) {
  const [presetHours, setPresetHours] = useState(24);
  const [customOpen, setCustomOpen] = useState(false);
  const [customFrom, setCustomFrom] = useState("");
  const [customTo, setCustomTo] = useState("");
  const [search, setSearch] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const pollRef = useRef(null);

  const { fromDate, toDate } = useMemo(() => {
    if (customOpen && customFrom && customTo) {
      return { fromDate: new Date(customFrom), toDate: new Date(customTo) };
    }
    const to = new Date();
    const from = new Date(to.getTime() - presetHours * 3600 * 1000);
    return { fromDate: from, toDate: to };
  }, [presetHours, customOpen, customFrom, customTo]);

  useEffect(() => {
    if (!customFrom) setCustomFrom(toLocalInputValue(new Date(Date.now() - 24 * 3600 * 1000)));
    if (!customTo) setCustomTo(toLocalInputValue(new Date()));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function load() {
    const params = new URLSearchParams({
      from_iso: fromDate.toISOString(),
      to_iso: toDate.toISOString(),
    });

    fetch(`${API_BASE}/dashboard/fqc?${params.toString()}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Request failed (${r.status})`);
        return r.json();
      })
      .then((d) => {
        setData(d);
        setError(null);
        setLastUpdated(new Date());
      })
      .catch((e) => setError(e.message));
  }

  useEffect(() => {
    load();
    clearInterval(pollRef.current);
    pollRef.current = setInterval(load, 20000);
    return () => clearInterval(pollRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fromDate.getTime(), toDate.getTime()]);

  useEffect(() => {
    if (refreshSignal) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshSignal]);

  const filteredStream = useMemo(() => {
    if (!data) return [];
    const q = search.trim().toLowerCase();
    if (!q) return data.alert_stream;
    return data.alert_stream.filter(
      (a) => a.narration_en.toLowerCase().includes(q) || a.alert_type.toLowerCase().includes(q)
    );
  }, [data, search]);

  const station = data?.by_zone?.[0];

  return (
    <div className="fqc-shell">
      <div className="fqc-topbar">
        <Logo />
        <div className="fqc-page-title">FQC Monitoring</div>
      </div>

      <div className="fqc-body">
        <div className="fqc-header-card">
          <div className="fqc-header-left">
            <span className="fqc-eyebrow">FQC INTELLIGENCE</span>
            <h1>FQC Station 1</h1>
            <p>Live alert counts and inspection timing from the safety alert database.</p>
          </div>

          <div className="fqc-time-window">
            <div className="fqc-time-label">Time Window</div>
            <div className="fqc-time-sub">
              {customOpen ? "Custom range" : `Last ${PRESETS.find((p) => p.hours === presetHours)?.label}`}
            </div>
            <div className="fqc-preset-row">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  className={`fqc-preset-btn ${!customOpen && presetHours === p.hours ? "active" : ""}`}
                  onClick={() => {
                    setCustomOpen(false);
                    setPresetHours(p.hours);
                  }}
                >
                  {p.label}
                </button>
              ))}
              <button
                className={`fqc-preset-btn ${customOpen ? "active" : ""}`}
                onClick={() => setCustomOpen(true)}
              >
                Custom
              </button>
            </div>
            {customOpen && (
              <div className="fqc-custom-range">
                <label>
                  FROM
                  <input type="datetime-local" value={customFrom} onChange={(e) => setCustomFrom(e.target.value)} />
                </label>
                <label>
                  TO
                  <input type="datetime-local" value={customTo} onChange={(e) => setCustomTo(e.target.value)} />
                </label>
              </div>
            )}
          </div>
        </div>

        <div className="fqc-filter-row">
          <input
            type="text"
            placeholder="Search narration / alert type"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {lastUpdated && (
            <span className="fqc-updated">Updated {lastUpdated.toLocaleTimeString()} · auto-refreshes every 20s</span>
          )}
        </div>

        {error && <div className="fqc-error">Couldn't reach the dashboard API: {error}</div>}

        {data && (
          <>
            <div className="fqc-stats-row">
              <StatCard label="Total Alerts" value={data.total_alerts} accent="#ee7d22" hint="in selected window" />
              <StatCard label="Hand Touch Alerts" value={data.by_type.HAND_TOUCH} accent="#dc2626" />
              <StatCard label="Fast Inspection Alerts" value={data.by_type.FAST_INSPECTION} accent="#d97706" />
              <StatCard label="Missing Cleaning Alerts" value={data.by_type.MISSING_CLEANING} accent="#1d5a96" />
              <StatCard
                label="Avg Inspection Time"
                value={data.avg_inspection_time != null ? `${data.avg_inspection_time}s` : "—"}
                accent="#1d5a96"
                hint={station?.last_alert_at ? `Last alert ${new Date(station.last_alert_at).toLocaleTimeString()}` : undefined}
              />
            </div>

            <div className="fqc-panel">
              <div className="fqc-panel-head">
                <h2>Alert Stream — FQC Station 1</h2>
                <p>Most recent alerts for the selected time window</p>
              </div>
              <div className="fqc-table-wrap">
                <table className="fqc-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Alert Type</th>
                      <th>Inspection (s)</th>
                      <th>Narration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredStream.length === 0 && (
                      <tr><td colSpan={4} className="fqc-empty">No alerts in this window.</td></tr>
                    )}
                    {filteredStream.map((a, i) => (
                      <tr key={i}>
                        <td className="fqc-time-cell">{new Date(a.timestamp).toLocaleString()}</td>
                        <td><TypePill type={a.alert_type} /></td>
                        <td>{a.inspection_time}</td>
                        <td className="fqc-narration-cell">{a.narration_en}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
