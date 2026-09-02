/** Hash-based routing.
 *
 * Hand-rolled rather than pulling in a router library. The route space here is
 * two segments deep -- `#/project/tab` -- and hash routing means the built site
 * works from any static host or a `file://` path with no server rewrite rules,
 * which matters for the screenshot capture and for anyone opening the dist
 * folder directly. Deep links still work, so a video can jump straight to a tab.
 */

import { useCallback, useEffect, useState } from "react";

export interface Route {
  page: "home" | "project" | "methods" | "datasets";
  slug?: string;
  tab?: string;
}

export function parseHash(hash: string): Route {
  const clean = hash.replace(/^#\/?/, "").split("?")[0];
  const parts = clean.split("/").filter(Boolean);
  if (!parts.length) return { page: "home" };
  if (parts[0] === "methods") return { page: "methods" };
  if (parts[0] === "datasets") return { page: "datasets" };
  return { page: "project", slug: parts[0], tab: parts[1] ?? "overview" };
}

export function href(route: Route): string {
  if (route.page === "home") return "#/";
  if (route.page === "methods") return "#/methods";
  if (route.page === "datasets") return "#/datasets";
  return `#/${route.slug}/${route.tab ?? "overview"}`;
}

export function useRoute(): [Route, (r: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parseHash(window.location.hash));

  useEffect(() => {
    const onChange = () => {
      setRoute(parseHash(window.location.hash));
      // A tab switch should start at the top of the new content, not halfway
      // down where the previous tab happened to be scrolled.
      window.scrollTo({ top: 0, behavior: "auto" });
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = useCallback((r: Route) => {
    window.location.hash = href(r);
  }, []);

  return [route, navigate];
}
