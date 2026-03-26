import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Zap } from "lucide-react"
import { useLiveData } from "@/hooks/useLiveData"

export function RecentAlerts() {
  const { flashes } = useLiveData()
  const recent = [...flashes].reverse().slice(0, 8)

  return (
    <Card className="col-span-1 lg:col-span-2 border-border bg-card flex flex-col h-full">
      <CardHeader className="flex-shrink-0 py-2 px-3">
        <CardTitle className="text-xs font-medium text-foreground">Alertes récentes</CardTitle>
        <CardDescription className="text-[10px] text-muted-foreground">
          {flashes.length > 0 ? `${flashes.length} impacts reçus` : "En attente d'impacts…"}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto p-0 px-3 pb-2">
        {recent.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">Aucun impact pour l'instant</p>
        ) : (
          <div className="space-y-2">
            {recent.map((flash) => (
              <div key={flash.rank} className="flex items-center py-1">
                <div
                  className="h-7 w-7 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{
                    backgroundColor: flash.flash_type === "CG" ? "var(--chart-2)" : "var(--chart-1)",
                    opacity: 0.85,
                  }}
                >
                  <Zap className="h-3.5 w-3.5 text-white" />
                </div>
                <div className="ml-2 space-y-0 flex-1 min-w-0">
                  <p className="text-xs font-medium leading-none text-foreground truncate">
                    {flash.lat.toFixed(4)}, {flash.lon.toFixed(4)}
                  </p>
                  <p className="text-[10px] text-muted-foreground">
                    {flash.dist_km.toFixed(1)} km •{" "}
                    <span className="font-semibold">{flash.flash_type}</span>
                  </p>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className="text-[10px] font-mono text-muted-foreground">
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
