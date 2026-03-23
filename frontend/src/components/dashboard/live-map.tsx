import {
  Map,
  MapMarker,
  MapPopup,
  MapTileLayer,
  MapZoomControl,
  MapCircle,
  MapLayerGroup,
  MapLayers,
  MapLayersControl,
  MapFullscreenControl,
  MapMarkerClusterGroup,
  MapTooltip,
} from "@/components/ui/map"
import { Zap } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

// coordinates for Bastia Airport -> todo: replace with dynamic data from API
const BASTIA_COORDS: [number, number] = [42.5527, 9.4837]

// mock data for recent impacts -> todo: same thing, replace with real API data
const recentImpacts = [
  { id: 1, pos: [42.56, 9.49] as [number, number], type: "IC", time: "14:02" },
  { id: 2, pos: [42.54, 9.47] as [number, number], type: "CG", time: "14:05" },
  { id: 3, pos: [42.57, 9.51] as [number, number], type: "IC", time: "14:08" },
  { id: 4, pos: [42.53, 9.44] as [number, number], type: "CG", time: "14:12" },
  { id: 5, pos: [42.55, 9.48] as [number, number], type: "CG", time: "14:15" },
]

export const LiveMap = () => {
  return (
    <div className="col-span-6 relative w-full h-[600px] rounded-xl overflow-hidden border border-border bg-card shadow-sm">
      <Map center={BASTIA_COORDS} zoom={11} className="z-0 h-full w-full">
        <MapLayers defaultLayerGroups={["Zones de sécurité", "Impacts récents"]}>
          <MapTileLayer name="Standard" />
          
          {/* Security Zones Group */}
          <MapLayerGroup name="Zones de sécurité">
            <MapCircle 
              center={BASTIA_COORDS} 
              radius={20000} 
              className="stroke-[var(--chart-2)] fill-[var(--chart-2)]"
            />
            <MapCircle 
              center={BASTIA_COORDS} 
              radius={50000} 
              className="stroke-muted-foreground fill-transparent stroke-1 dash-array-[10,10]"
            />
            <MapMarker position={BASTIA_COORDS}>
              <MapTooltip side="top">Aéroport de Bastia</MapTooltip>
            </MapMarker>
          </MapLayerGroup>

          {/* Lightning Impacts Group */}
          <MapLayerGroup name="Impacts récents">
            <MapMarkerClusterGroup>
              {recentImpacts.map((impact) => (
                <MapMarker 
                  key={impact.id} 
                  position={impact.pos}
                  icon={
                    <div className="relative group">
                      <Zap 
                        className="size-6 transition-transform group-hover:scale-125" 
                        style={{ 
                          color: impact.type === 'CG' ? 'var(--chart-2)' : 'var(--chart-1)',
                          fill: impact.type === 'CG' ? 'var(--chart-2)' : 'var(--chart-1)',
                          filter: 'drop-shadow(0 0 4px rgba(0,0,0,0.5))'
                        }} 
                      />
                    </div>
                  }
                >
                  <MapPopup>
                    <div className="flex flex-col gap-1 text-center">
                      <p className="font-bold text-sm">Impact {impact.type === 'CG' ? 'Nuage-Sol' : 'Intra-Nuage'}</p>
                      <p className="text-xs text-muted-foreground">Heure : {impact.time} UTC</p>
                      <p className="text-xs text-muted-foreground">Coord : {impact.pos[0]}, {impact.pos[1]}</p>
                    </div>
                  </MapPopup>
                </MapMarker>
              ))}
            </MapMarkerClusterGroup>
          </MapLayerGroup>

          {/* Map controls */}
          <MapLayersControl position="top-1 right-1" className="mt-2 mr-2" />
          <MapFullscreenControl position="top-1 left-1" className="mt-2 ml-2" />
          <MapZoomControl position="bottom-1 right-1" className="mb-2 mr-2" />  
        </MapLayers>
      </Map>

      {/* Legend overlay */}
      <Card className="absolute bottom-4 left-4 z-[1000] w-fit bg-card/80 backdrop-blur-md border-border shadow-md pointer-events-none">
        <CardContent className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="size-3 rounded-full border-2 border-[var(--chart-2)] bg-[var(--chart-2)]/20" />
            <span className="text-[10px] font-medium text-foreground uppercase tracking-wider">Alerte (20km)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-3 border-t-2 border-dashed border-muted-foreground w-4" />
            <span className="text-[10px] font-medium text-foreground uppercase tracking-wider">Surveillance (50km)</span>
          </div>
          <Separator className="bg-border" />
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <Zap className="size-3 text-[var(--chart-2)] fill-[var(--chart-2)]" />
              <span className="text-[10px] font-bold text-foreground">CG</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Zap className="size-3 text-[var(--chart-1)] fill-[var(--chart-1)]" />
              <span className="text-[10px] font-bold text-foreground">IC</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
