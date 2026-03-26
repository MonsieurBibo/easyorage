import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useLiveData } from "@/hooks/useLiveData"
import { Clock, Zap, CheckCircle2, AlertCircle } from "lucide-react"

export function Reports() {
  const { alertsHistory } = useLiveData()

  if (!alertsHistory || alertsHistory.length === 0) {
    return (
      <div className="flex items-center justify-center h-[400px] text-muted-foreground">
        Aucun rapport disponible pour cet aéroport.
      </div>
    )
  }

  return (
    <Card className="border-border bg-card">
      <CardHeader>
        <CardTitle>Historique des alertes</CardTitle>
        <CardDescription>Liste des épisodes orageux passés et performance du modèle</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="relative w-full overflow-auto">
          <table className="w-full caption-bottom text-sm">
            <thead className="[&_tr]:border-b border-border">
              <tr className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">ID</th>
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Début</th>
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Durée</th>
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Éclairs</th>
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Prédiction</th>
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Status</th>
              </tr>
            </thead>
            <tbody className="[&_tr:last-child]:border-0">
              {alertsHistory.map((alert) => {
                const startDate = new Date(alert.start_date)
                const durationMin = Math.round(alert.duration_s / 60)
                
                return (
                  <tr key={alert.alert_id} className="border-b transition-colors hover:bg-muted/50 border-border">
                    <td className="p-4 align-middle font-mono font-bold text-xs">#{alert.alert_id}</td>
                    <td className="p-4 align-middle">
                      <div className="flex flex-col">
                        <span className="font-medium">{startDate.toLocaleDateString("fr-FR")}</span>
                        <span className="text-xs text-muted-foreground">
                          {startDate.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      </div>
                    </td>
                    <td className="p-4 align-middle">
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3 text-muted-foreground" />
                        {durationMin} min
                      </div>
                    </td>
                    <td className="p-4 align-middle">
                      <div className="flex items-center gap-1">
                        <Zap className="h-3 w-3 text-chart-1" />
                        {alert.n_flashes}
                      </div>
                    </td>
                    <td className="p-4 align-middle">
                      {alert.prediction ? (
                        <div className="flex flex-col">
                          <span className="font-medium text-chart-2">{(alert.prediction.confidence * 100).toFixed(0)}%</span>
                          <span className="text-[10px] text-muted-foreground">Flash #{alert.prediction.triggered_at_rank}</span>
                        </div>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </td>
                    <td className="p-4 align-middle text-right">
                      {alert.prediction ? (
                        <div className="flex items-center gap-1.5 text-chart-2">
                          <CheckCircle2 className="h-4 w-4" />
                          <span className="text-xs font-semibold uppercase">Couvert</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 text-muted-foreground">
                          <AlertCircle className="h-4 w-4" />
                          <span className="text-xs font-semibold uppercase">Ignoré</span>
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
