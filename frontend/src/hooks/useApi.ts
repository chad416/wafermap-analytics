import { useEffect, useState } from "react";

export interface ApiState<T> {
  data: T | undefined;
  error: string | undefined;
  loading: boolean;
}

interface Settled<T> {
  key: string | null;
  data?: T;
  error?: string;
}

/**
 * Run `fetcher` whenever `deps` change. The previous request is aborted so a slow response
 * can never overwrite a newer one, and the last data stays visible while the next page loads.
 * `loading` is derived (settled key !== current key), so the effect only sets state from the
 * async callback.
 */
export function useApi<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
): ApiState<T> {
  const key = JSON.stringify(deps);
  const [settled, setSettled] = useState<Settled<T>>({ key: null });

  useEffect(() => {
    const ctrl = new AbortController();
    fetcher(ctrl.signal)
      .then((data) => setSettled({ key, data }))
      .catch((err: unknown) => {
        if (ctrl.signal.aborted) return;
        setSettled({ key, error: err instanceof Error ? err.message : String(err) });
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `key` encodes the caller's deps
  }, [key]);

  const loading = settled.key !== key;
  return { data: settled.data, error: loading ? undefined : settled.error, loading };
}
