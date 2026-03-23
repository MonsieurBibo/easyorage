import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Map as MapIcon } from "lucide-react"

export function LiveMap() {
  return (
    <Card className="col-span-4 min-h-[500px] flex flex-col border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base font-medium text-foreground">Carte en direct</CardTitle>
        <MapIcon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent className="flex-1 flex items-center justify-center bg-muted/30 border-2 border-dashed border-border rounded-md m-4">
        <div className="text-center text-muted-foreground">
          <MapIcon className="h-12 w-12 mx-auto mb-2 opacity-20" style={{ color: 'var(--primary)' }} />
          <p className="font-medium">Intégration Leaflet / Mapbox</p>
          <p className="text-xs">Visualisation spatio-temporelle des impacts (20km/50km)</p>
        </div>
      </CardContent>
    </Card>
  )
}
