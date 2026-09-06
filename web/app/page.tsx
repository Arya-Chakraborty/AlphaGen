"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Stats = {
  min: number;
  max: number;
  p10?: number;
  p90?: number;
  n?: number;
};

type Selected = {
  name: string;
  min: string;
  max: string;
  stats?: Stats;
};

type Candidate = {
  canonical_smiles?: string;
  penalty?: number;
  in_range?: boolean;
  n_passed?: number;
  generation?: number;
  origin?: string;
  values?: Record<string, number | null>;
};

type Hit = Candidate & {
  inchikey?: string;
  passed?: Record<string, boolean>;
};

type EventMsg = {
  type: string;
  message?: string;
  generation?: number;
  n_hits?: number;
  n_requested?: number;
  n_population?: number;
  best_penalty?: number | null;
  best?: Candidate | null;
  population?: Candidate[];
  hits?: Hit[];
  manifest?: Record<string, unknown>;
};

function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Math.abs(value) >= 100) return value.toFixed(1);
  return value.toFixed(digits);
}

function apiError(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item
          ? String((item as { msg: string }).msg)
          : JSON.stringify(item),
      )
      .join("; ");
  }
  return fallback;
}

function depictUrl(smiles: string): string {
  return `${API}/depict?smiles=${encodeURIComponent(smiles)}`;
}

export default function HomePage() {
  const [allNames, setAllNames] = useState<string[]>([]);
  const [featured, setFeatured] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Selected[]>([]);
  const [n, setN] = useState(5);
  const [generations, setGenerations] = useState(20);
  const [population, setPopulation] = useState(64);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [generation, setGeneration] = useState<EventMsg | null>(null);
  const [hits, setHits] = useState<Hit[]>([]);
  const logRef = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    fetch(`${API}/descriptors`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json();
      })
      .then((data) => {
        setAllNames(data.names ?? []);
        setFeatured(data.featured ?? []);
      })
      .catch((err: Error) => setError(`Cannot reach backend at ${API}: ${err.message}`));
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log]);

  const suggestions = useMemo(() => {
    const taken = new Set(selected.map((row) => row.name));
    const q = query.trim().toLowerCase();
    const pool = q
      ? allNames.filter((name) => name.toLowerCase().includes(q))
      : featured;
    return pool.filter((name) => !taken.has(name)).slice(0, 12);
  }, [allNames, featured, query, selected]);

  async function addDescriptor(name: string) {
    if (selected.some((row) => row.name === name)) return;
    setQuery("");
    setSelected((rows) => [...rows, { name, min: "", max: "" }]);
    try {
      const res = await fetch(`${API}/descriptors/stats?names=${encodeURIComponent(name)}`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const stats = data.stats?.[name] as Stats | undefined;
      setSelected((rows) =>
        rows.map((row) => (row.name === name ? { ...row, stats } : row)),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function updateRange(name: string, field: "min" | "max", value: string) {
    setSelected((rows) =>
      rows.map((row) => (row.name === name ? { ...row, [field]: value } : row)),
    );
  }

  function removeDescriptor(name: string) {
    setSelected((rows) => rows.filter((row) => row.name !== name));
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    if (selected.length === 0) {
      setError("Select at least one descriptor.");
      return;
    }
    const descriptors: Record<string, { min: number; max: number }> = {};
    for (const row of selected) {
      if (row.min === "" || row.max === "") {
        setError(`Enter min and max for ${row.name}.`);
        return;
      }
      const min = Number(row.min);
      const max = Number(row.max);
      if (!Number.isFinite(min) || !Number.isFinite(max)) {
        setError(`${row.name}: min and max must be numbers.`);
        return;
      }
      if (min > max) {
        setError(`${row.name}: min is greater than max.`);
        return;
      }
      descriptors[row.name] = { min, max };
    }

    setRunning(true);
    setLog([]);
    setHits([]);
    setGeneration(null);

    try {
      const res = await fetch(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n, descriptors, population, generations, seed: 0 }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(apiError(body, res.statusText));
      }
      const jobId = body.id as string;
      await listen(jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setRunning(false);
    }
  }

  function listen(jobId: string): Promise<void> {
    return new Promise((resolve) => {
      const source = new EventSource(`${API}/generate/${jobId}/events`);
      source.onmessage = (msg) => {
        const event = JSON.parse(msg.data) as EventMsg;
        if (event.message) {
          setLog((lines) => [...lines, event.message as string]);
        }
        if (event.type === "generation" || event.type === "ga_start") {
          setGeneration(event);
          if (event.hits) setHits(event.hits);
        }
        if (event.type === "done") {
          if (event.hits) setHits(event.hits);
          source.close();
          setRunning(false);
          resolve();
        }
        if (event.type === "error") {
          setError(event.message || "Generation failed");
          source.close();
          setRunning(false);
          resolve();
        }
      };
      source.onerror = () => {
        source.close();
        setRunning(false);
        setError((prev) => prev || "Lost connection to the backend.");
        resolve();
      };
    });
  }

  return (
    <main className="page">
      <header className="hero">
        <p className="kicker">molecule-gen</p>
        <h1>Generate molecules from Mordred ranges</h1>
        <p className="lede">
          Pick descriptors, type the ranges you want, then the SELFIES GPT proposes
          seeds and the genetic algorithm steers them into the box. Papyrus min/max
          is a reference only — your ranges can sit outside the corpus.
        </p>
      </header>

      <div className="grid">
        <form className="panel" onSubmit={onSubmit}>
          <h2>Spec</h2>
          <label className="field">
            Search descriptors
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="MW, SLogP, TopoPSA…"
            />
          </label>
          <div className="chips">
            {suggestions.map((name) => (
              <button
                key={name}
                type="button"
                className="chip"
                onClick={() => addDescriptor(name)}
              >
                {name}
              </button>
            ))}
          </div>

          {selected.length === 0 ? (
            <p className="muted">No descriptors yet. Add a few from the list above.</p>
          ) : (
            <ul className="ranges">
              {selected.map((row) => (
                <li key={row.name}>
                  <div className="range-head">
                    <strong>{row.name}</strong>
                    <button type="button" className="link" onClick={() => removeDescriptor(row.name)}>
                      remove
                    </button>
                  </div>
                  <p className="ref">
                    {row.stats
                      ? `Papyrus corpus: ${fmt(row.stats.min)} – ${fmt(row.stats.max)} (reference, not a limit)`
                      : "Loading corpus range…"}
                  </p>
                  <div className="pair">
                    <label>
                      min
                      <input
                        inputMode="decimal"
                        value={row.min}
                        onChange={(e) => updateRange(row.name, "min", e.target.value)}
                        placeholder={row.stats ? fmt(row.stats.min) : ""}
                      />
                    </label>
                    <label>
                      max
                      <input
                        inputMode="decimal"
                        value={row.max}
                        onChange={(e) => updateRange(row.name, "max", e.target.value)}
                        placeholder={row.stats ? fmt(row.stats.max) : ""}
                      />
                    </label>
                  </div>
                </li>
              ))}
            </ul>
          )}

          <div className="pair">
            <label>
              molecules
              <input
                type="number"
                min={1}
                max={50}
                value={n}
                onChange={(e) => setN(Number(e.target.value))}
              />
            </label>
            <label>
              generations
              <input
                type="number"
                min={0}
                max={200}
                value={generations}
                onChange={(e) => setGenerations(Number(e.target.value))}
              />
            </label>
            <label>
              population
              <input
                type="number"
                min={4}
                max={256}
                value={population}
                onChange={(e) => setPopulation(Number(e.target.value))}
              />
            </label>
          </div>

          <button className="go" type="submit" disabled={running}>
            {running ? "Generating…" : "Generate"}
          </button>
          {error ? <p className="error">{error}</p> : null}
        </form>

        <section className="panel">
          <h2>Genetic algorithm</h2>
          <pre className="log" ref={logRef}>
            {log.length ? log.join("\n") : "Backend progress will stream here."}
          </pre>
          {generation?.population?.length ? (
            <div>
              <h3>
                Generation {generation.generation ?? 0} · hits {generation.n_hits ?? 0}/
                {generation.n_requested ?? n}
              </h3>
              <div className="mols">
                {generation.population.slice(0, 6).map((mol, idx) => (
                  <article key={`${mol.canonical_smiles}-${idx}`} className="mol">
                    {mol.canonical_smiles ? (
                      <img src={depictUrl(mol.canonical_smiles)} alt={mol.canonical_smiles} />
                    ) : null}
                    <p className="smiles">{mol.canonical_smiles}</p>
                    <p className="muted">
                      penalty {fmt(mol.penalty, 4)} · passed {mol.n_passed ?? 0}
                    </p>
                  </article>
                ))}
              </div>
            </div>
          ) : null}
        </section>
      </div>

      <section className="panel results">
        <h2>In-range molecules</h2>
        {hits.length === 0 ? (
          <p className="muted">Hits will appear here when a molecule satisfies every range.</p>
        ) : (
          <div className="mols">
            {hits.map((hit) => (
              <article key={hit.inchikey || hit.canonical_smiles} className="mol hit">
                {hit.canonical_smiles ? (
                  <img src={depictUrl(hit.canonical_smiles)} alt={hit.canonical_smiles} />
                ) : null}
                <p className="smiles">{hit.canonical_smiles}</p>
                <p className="muted">
                  gen {hit.generation ?? 0} · {hit.origin} · penalty {fmt(hit.penalty, 4)}
                </p>
                <ul className="vals">
                  {Object.entries(hit.values || {}).map(([name, value]) => (
                    <li key={name}>
                      <span>{name}</span>
                      <strong className={hit.passed?.[name] ? "ok" : ""}>{fmt(value)}</strong>
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
