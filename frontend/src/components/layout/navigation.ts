import { BookOpen, Gauge, Globe, Images, Info, ScanSearch, Satellite } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export interface NavItem {
  id: string;
  path: string;
  label: string;
  icon: LucideIcon;
  /** "live": implemented in this build. "planned": placeholder page that explains what is missing. */
  status: "live" | "planned";
  summary: string;
  needs: string[];
}

export const NAV_ITEMS: NavItem[] = [
  {
    id: "overview",
    path: "/",
    label: "Overview",
    icon: Globe,
    status: "live",
    summary: "",
    needs: [],
  },
  {
    id: "analyze",
    path: "/analyze",
    label: "Analyze",
    icon: ScanSearch,
    status: "planned",
    summary:
      "Pick a sample or upload a four-band GeoTIFF, run the model, and inspect the cloud mask, probability map and coverage statistics.",
    needs: ["A trained U-Net checkpoint", "The prediction API (/api/v1/predict)"],
  },
  {
    id: "samples",
    path: "/samples",
    label: "Sample Library",
    icon: Images,
    status: "planned",
    summary:
      "Curated 38-Cloud patches, each with the four-band input the model needs and its ground-truth mask for comparison.",
    needs: ["Selected demonstration patches with attribution", "The samples API (/api/v1/samples)"],
  },
  {
    id: "performance",
    path: "/performance",
    label: "Model Performance",
    icon: Gauge,
    status: "planned",
    summary:
      "Measured results on the 20 official test scenes: IoU, Dice, precision, recall, pixel accuracy and training curves.",
    needs: ["A completed training run", "Scene-level evaluation artifacts in artifacts/metrics"],
  },
  {
    id: "mission",
    path: "/mission",
    label: "Mission & Data",
    icon: Satellite,
    status: "planned",
    summary:
      "Landsat 8, the 38-Cloud dataset and the spectral bands the model uses, with references to NASA, USGS and the dataset authors.",
    needs: ["Verified source references"],
  },
  {
    id: "methodology",
    path: "/methodology",
    label: "Methodology",
    icon: BookOpen,
    status: "planned",
    summary:
      "How the U-Net, the BCE + Dice loss and the evaluation metrics work, and where the approach has limits.",
    needs: ["The final training configuration and results"],
  },
  {
    id: "about",
    path: "/about",
    label: "About",
    icon: Info,
    status: "planned",
    summary: "Project background, team credits and known limitations.",
    needs: ["Team details"],
  },
];
