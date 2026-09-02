import type { ProjectMeta } from "../../lib/types";
import { Callout } from "../../components/ui";
import TaxiLive from "../live/TaxiLive";
import ChurnLive from "../live/ChurnLive";
import SegmentsLive from "../live/SegmentsLive";
import BasketLive from "../live/BasketLive";
import AnomalyLive from "../live/AnomalyLive";
import ForecastLive from "../live/ForecastLive";
import AutomlLive from "../live/AutomlLive";
import NanollmLive from "../live/NanollmLive";

export default function LiveTab({ slug, meta }: { slug: string; meta: ProjectMeta }) {
  switch (slug) {
    case "taxi":
      return <TaxiLive />;
    case "churn":
      return <ChurnLive />;
    case "segments":
      return <SegmentsLive />;
    case "basket":
      return <BasketLive />;
    case "anomaly":
      return <AnomalyLive />;
    case "forecast":
      return <ForecastLive />;
    case "automl":
      return <AutomlLive />;
    case "nanollm":
      return <NanollmLive />;
    default:
      return (
        <Callout tone="info" title="No interactive demo">
          {meta.title} does not expose a live inference playground.
        </Callout>
      );
  }
}
