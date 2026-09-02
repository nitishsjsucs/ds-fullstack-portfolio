/** Thin fetch wrapper plus a `useApi` hook.
 *
 * Two behaviours worth naming. First, the backend returns 503 with a helpful
 * message when a project has no artifacts yet, so the UI can say "run
 * train_all.py" rather than showing a generic failure. Second, GET responses are
 * memoised for the session: every payload is a pinned artifact of one training
 * run, so re-fetching on tab switches would be pure waste.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const BASE = "/api";
const cache = new Map<string, unknown>();

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly hint?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parse(res: Response): Promise<unknown> {
  if (res.ok) return res.json();
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    /* non-JSON error body; keep the status line */
  }
  const hint =
    res.status === 503
      ? "This project has not been trained yet. Run `python scripts/train_all.py` from the repository root."
      : res.status === 404
        ? "The API has no payload at this address."
        : undefined;
  throw new ApiError(detail, res.status, hint);
}

export async function apiGet<T>(path: string, { fresh = false } = {}): Promise<T> {
  if (!fresh && cache.has(path)) return cache.get(path) as T;
  const res = await fetch(`${BASE}${path}`);
  const data = await parse(res);
  cache.set(path, data);
  return data as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return (await parse(res)) as T;
}

export function clearCache(): void {
  cache.clear();
}

type State<T> =
  | { status: "loading"; data: null; error: null }
  | { status: "ready"; data: T; error: null }
  | { status: "error"; data: null; error: ApiError };

/** Fetch a GET endpoint, tracking loading and error state. */
export function useApi<T>(path: string | null): State<T> & { reload: () => void } {
  const [state, setState] = useState<State<T>>({
    status: "loading",
    data: null,
    error: null,
  });
  // Guards against a slow response for a previous path overwriting a newer one
  // when the user switches project or tab mid-flight.
  const latest = useRef(0);

  const load = useCallback(
    (fresh = false) => {
      if (!path) return;
      const ticket = ++latest.current;
      setState({ status: "loading", data: null, error: null });
      apiGet<T>(path, { fresh })
        .then((data) => {
          if (ticket === latest.current) setState({ status: "ready", data, error: null });
        })
        .catch((err: unknown) => {
          if (ticket !== latest.current) return;
          const error =
            err instanceof ApiError
              ? err
              : new ApiError(
                  err instanceof Error ? err.message : "Network request failed",
                  0,
                  "Is the backend running? Start it with `make api` or `uvicorn app.main:app`.",
                );
          setState({ status: "error", data: null, error });
        });
    },
    [path],
  );

  useEffect(() => {
    load(false);
  }, [load]);

  return { ...state, reload: () => load(true) };
}

/** POST-driven state for the inference playgrounds. */
export function useMutation<TBody, TResult>(path: string) {
  const [data, setData] = useState<TResult | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [pending, setPending] = useState(false);
  const latest = useRef(0);

  const run = useCallback(
    async (body: TBody) => {
      const ticket = ++latest.current;
      setPending(true);
      setError(null);
      try {
        const result = await apiPost<TResult>(path, body);
        if (ticket === latest.current) setData(result);
        return result;
      } catch (err) {
        if (ticket === latest.current) {
          setError(
            err instanceof ApiError
              ? err
              : new ApiError(err instanceof Error ? err.message : "Request failed", 0),
          );
        }
        return null;
      } finally {
        if (ticket === latest.current) setPending(false);
      }
    },
    [path],
  );

  return { data, error, pending, run };
}
