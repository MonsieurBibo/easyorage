import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Zap, Clock, AlertTriangle, PlaneTakeoff } from "lucide-react"
import { DashboardHeader } from "./dashboard/header"
import { StatCard } from "./dashboard/stat-card"
import { LiveMap } from "./dashboard/live-map"
import { RecentAlerts } from "./dashboard/recent-alerts"
import { useLiveData } from "@/context/LiveDataContext"

const SPEEDS = [0.5, 1, 5, 10, 30, 100]

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

  // Stat card values
  const scoreStr = currentFlash ? `${(currentFlash.score * 100).toFixed(0)}%` : "—"
  const impacts20 = flashes.filter((f) => f.dist_km < 20).length
  const statusValue = alertEnded ? "TERMINÉE" : isReplaying ? "EN ALERTE" : "—"
  const statusDesc = alertMeta
    ? alertEnded
      ? `Alerte terminée · ${alertMeta.n_flashes} éclairs`
      : `En cours · ${flashes.length} / ${alertMeta.n_flashes} éclairs`
    : "En attente de données"

  const predValue = prediction
    ? `${(prediction.confidence * 100).toFixed(0)}%`
    : "—"
  const predDesc = prediction
    ? `Fin détectée au flash #${prediction.triggered_at_rank}`
    : "Non détectée"

  const airportLabel = currentAirport?.name ?? selectedAirport

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      <DashboardHeader />

      <main className="flex-1 space-y-4 p-8 pt-6">
        <div className="flex items-center justify-between">
          <h2 className="text-3xl font-bold tracking-tight">Dashboard — {airportLabel}</h2>

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

        <Tabs defaultValue="overview" className="space-y-4">
          <TabsList className="bg-muted border border-border">
            <TabsTrigger value="overview" className="data-[state=active]:bg-card data-[state=active]:text-foreground text-muted-foreground">
              Carte en direct
            </TabsTrigger>
            <TabsTrigger value="analytics" className="data-[state=active]:bg-card data-[state=active]:text-foreground text-muted-foreground">
              Statistiques
            </TabsTrigger>
            <TabsTrigger value="reports" className="data-[state=active]:bg-card data-[state=active]:text-foreground text-muted-foreground">
              Rapports d'alerte
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <StatCard
                title="Probabilité de fin"
                value={scoreStr}
                description="Score du dernier éclair"
                icon={Zap}
                trendColor="var(--chart-1)"
              />
              <StatCard
                title="Fin d'alerte"
                value={predValue}
                description={predDesc}
                icon={Clock}
                trendColor="var(--chart-2)"
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
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
