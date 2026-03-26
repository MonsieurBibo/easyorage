import { Zap, Clock, AlertTriangle, PlaneTakeoff } from "lucide-react"
import { StatCard } from "./dashboard/stat-card"
import { LiveMap } from "./dashboard/live-map"
import { RecentAlerts } from "./dashboard/recent-alerts"
import { useLiveData } from "@/hooks/useLiveData"
import type { ChartConfig } from "@/components/ui/chart"
import { useMemo } from "react"

const SPEEDS = [0.5, 1, 5, 10, 30, 100]

// Chart configs for the mini charts
const probabilityChartConfig = {
  score: {
    label: "Probabilité",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig

export function Dashboard() {
  const {
    selectedAirport,
    currentAirport,
    isReplaying,
    alertMeta,
    flashes,
    currentFlash,
    prediction,
    alertEnded,
    speed,
    setSpeed,
  } = useLiveData()

  // Track probability history in real-time: score of each flash
  // Derived from flashes array
  const probabilityHistory = useMemo(() => {
    return flashes.map((flash, index) => ({
      date: `${index + 1}`,
      score: Math.round(flash.score * 100)
    }))
  }, [flashes])

  // Stat card values
  const scoreStr = currentFlash ? `${(currentFlash.score * 100).toFixed(0)}%` : "-"
  const impacts20 = flashes.filter((f) => f.dist_km < 20).length
  const statusValue = alertEnded ? "TERMINÉE" : isReplaying ? "EN ALERTE" : "-"
  const statusDesc = alertMeta
    ? alertEnded
      ? `Alerte terminée · ${alertMeta.n_flashes} éclairs`
      : `En cours · ${flashes.length} / ${alertMeta.n_flashes} éclairs`
    : "En attente de données"

  const predValue = prediction
    ? `${(prediction.confidence * 100).toFixed(0)}%`
    : "-"
  const predDesc = prediction
    ? `Fin détectée au flash #${prediction.triggered_at_rank}`
    : "Non détectée"

  const airportLabel = currentAirport?.name ?? selectedAirport

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-3xl font-bold tracking-tight">Dashboard - {airportLabel}</h2>

        {/* Speed selector */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Vitesse :</span>
          {SPEEDS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`px-2.5 py-1 rounded text-xs font-mono border transition-colors ${
                speed === s
                  ? "bg-primary text-primary-foreground border-primary"
                  : "border-border text-muted-foreground hover:bg-muted"
              }`}
            >
              {s}×
            </button>
          ))}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Probabilité de fin"
          value={scoreStr}
          description="Score du dernier éclair"
          icon={Zap}
          trendColor="var(--chart-1)"
          chartData={probabilityHistory}
          chartConfig={probabilityChartConfig}
          chartDataKey="score"
        />
        <StatCard
          title="Fin d'alerte"
          value={predValue}
          description={predDesc}
          icon={Clock}
          trendColor="var(--chart-2)"
          chartDataKey="confidence"
        />
        <StatCard
          title="Impacts (20km)"
          value={String(impacts20)}
          description={`Sur ${flashes.length} éclairs reçus`}
          icon={AlertTriangle}
          trendColor="var(--chart-3)"
        />
        <StatCard
          title="État de l'aéroport"
          value={statusValue}
          description={statusDesc}
          icon={PlaneTakeoff}
          trendColor="var(--chart-4)"
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-8">
        <LiveMap />
        <RecentAlerts />
      </div>
    </div>
  )
}
