import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Zap } from "lucide-react"
import { useLiveData } from "@/context/LiveDataContext"

export function RecentAlerts() {
  const { flashes } = useLiveData()
  const recent = [...flashes].reverse().slice(0, 8)

  return (
    <Card className="col-span-2 border-border bg-card">
      <CardHeader>
        <CardTitle className="text-base font-medium text-foreground">Alertes récentes</CardTitle>
        <CardDescription className="text-muted-foreground">
          {flashes.length > 0 ? `${flashes.length} impacts reçus` : "En attente d'impacts…"}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {recent.length === 0 ? (
          <p className="text-sm text-muted-foreground text-center py-8">Aucun impact pour l'instant</p>
        ) : (
          <div className="space-y-4">
            {recent.map((flash) => (
              <div key={flash.rank} className="flex items-center">
                <div
                  className="h-9 w-9 rounded-full flex items-center justify-center"
                  style={{
                    backgroundColor: flash.flash_type === "CG" ? "var(--chart-2)" : "var(--chart-1)",
                    opacity: 0.85,
                  }}
                >
                  <Zap className="h-4 w-4 text-white" />
                </div>
                <div className="ml-4 space-y-0.5 flex-1 min-w-0">
                  <p className="text-sm font-medium leading-none text-foreground">
                    {flash.lat.toFixed(4)}, {flash.lon.toFixed(4)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {flash.dist_km.toFixed(1)} km •{" "}
                    <span className="font-semibold">{flash.flash_type}</span>
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-xs font-mono text-muted-foreground">
                    {new Date(flash.date).toLocaleTimeString("fr-FR", { timeZone: "UTC", hour: "2-digit", minute: "2-digit" })}
                  </p>
                  <p className="text-xs font-semibold" style={{ color: flash.score > 0.5 ? "var(--chart-1)" : "var(--muted-foreground)" }}>
                    {(flash.score * 100).toFixed(0)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
