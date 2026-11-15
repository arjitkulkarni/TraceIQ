import { useEffect, useState } from "react";

export async function getJson(path) {
  const response = await fetch(`/api${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

/** Fetch a path. Pass null to skip. Stale responses are ignored. */
export function useFetch(path) {
  const [state, setState] = useState({ data: null, error: null, loading: Boolean(path) });
  useEffect(() => {
    if (!path) {
      setState({ data: null, error: null, loading: false });
      return undefined;
    }
    let current = true;
    setState((previous) => ({ ...previous, error: null, loading: true }));
    getJson(path)
      .then((data) => current && setState({ data, error: null, loading: false }))
      .catch((error) => current && setState({ data: null, error, loading: false }));
    return () => {
      current = false;
    };
  }, [path]);
  return state;
}

export const fmt = {
  count: (value) => Number(value).toLocaleString("en-US"),
  score: (value) => Number(value).toFixed(3),
  percent: (value) => `${(Number(value) * 100).toFixed(1)}%`,
};

export const MODEL_NAMES = {
  logistic: "Logistic Regression",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
  mlp: "MLP (no edges)",
  graphsage: "GraphSAGE",
  static_gcn: "Static GCN",
  evolvegcn_o: "EvolveGCN-O",
  evolvegcn_h: "EvolveGCN-H",
};
