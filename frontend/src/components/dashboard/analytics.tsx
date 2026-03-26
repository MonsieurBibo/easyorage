import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useLiveData } from "@/hooks/useLiveData"
import { Clock, Target, TrendingUp } from "lucide-react"

export function Analytics() {
  const { stats, alertsHistory } = useLiveData()

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-[400px] text-muted-foreground">
        Chargement des statistiques...
      </div>
    )
  }

  // Trier les alertes par date pour la timeline
  const sortedAlerts = [...alertsHistory].sort((a, b) => new Date(a.start_date).getTime() - new Date(b.start_date).getTime())

  return (
    <div className="space-y-6">
      {/* KPIs */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Gain moyen</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(stats.total_gain_h * 60 / Math.max(stats.total_alerts, 1)).toFixed(1)} min</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Couverture</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(stats.coverage_rate * 100).toFixed(0)} %</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Risque résiduel</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.risk} %</div>
          </CardContent>
        </Card>
      </div>

      {/* Timeline */}
      <Card>
        <CardHeader>
          <CardTitle>Chronologie de l'efficacité (dernières alertes)</CardTitle>
          <CardDescription>Visualisation de la détection et des impacts</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative border-l border-muted ml-2">
            {sortedAlerts.slice(-10).map((alert) => {
              const hasPrediction = !!alert.prediction
              return (
                <div key={alert.alert_id} className="mb-6 ml-6">
                  <div className={`absolute -left-2 mt-1.5 size-4 rounded-full border border-background ${hasPrediction ? "bg-chart-2" : "bg-muted"}`} />
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-semibold">{new Date(alert.start_date).toLocaleDateString()}</h4>
                      <p className="text-sm text-muted-foreground">
                        {alert.n_flashes} éclairs · {hasPrediction ? "Détecté" : "Non détecté"}
                      </p>
                    </div>
                    {hasPrediction && (
                      <span className="text-xs font-mono bg-chart-2/10 text-chart-2 px-2 py-1 rounded">
                        +{((new Date(alert.end_date).getTime() - new Date(alert.prediction!.triggered_at_date).getTime()) / 60000).toFixed(0)} min
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
