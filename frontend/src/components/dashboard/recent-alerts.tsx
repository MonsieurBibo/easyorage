import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Zap } from "lucide-react"

interface Alert {
  coord: string
  loc: string
  time: string
  type: "CG" | "IC"
}

const alerts: Alert[] = [
  { coord: "42.5527, 9.4837", loc: "Bastia, Ville", time: "il y a 4 min", type: "CG" },
  { coord: "42.5610, 9.4920", loc: "Poretta Nord", time: "il y a 7 min", type: "IC" },
  { coord: "42.5400, 9.4700", loc: "Zone Aéroportuaire", time: "il y a 12 min", type: "CG" },
  { coord: "42.5800, 9.5000", loc: "Mer Tyrrhénienne", time: "il y a 15 min", type: "IC" },
  { coord: "42.5200, 9.4500", loc: "Lucciana", time: "il y a 22 min", type: "CG" },
]

export function RecentAlerts() {
  return (  
    <Card className="col-span-2 border-border bg-card">
      <CardHeader>
        <CardTitle className="text-base font-medium text-foreground">Alertes récentes</CardTitle>
        <CardDescription className="text-muted-foreground">
          Derniers points d’impacts détectés
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {alerts.map((alert, i) => (
            <div key={i} className="flex items-center">
              <div 
                className="h-9 w-9 rounded-full flex items-center justify-center bg-muted"
                style={{ 
                  backgroundColor: alert.type === 'CG' ? 'var(--chart-2)' : 'var(--chart-1)',
                  opacity: 0.2
                }}
              >
                <Zap 
                  className="h-4 w-4" 
                  style={{ color: alert.type === 'CG' ? 'var(--chart-2)' : 'var(--chart-1)' }}
                />
              </div>
              <div className="ml-4 space-y-1 flex-1">
                <p className="text-sm font-medium leading-none text-foreground">{alert.coord}</p>
                <p className="text-sm text-muted-foreground">
                  {alert.loc} • <span className="font-semibold">{alert.type}</span>
                </p>
              </div>
              <div className="text-sm font-medium text-muted-foreground">{alert.time}</div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
