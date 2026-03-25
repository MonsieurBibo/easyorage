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
import { useLiveData } from "@/context/LiveDataContext"

export const LiveMap = () => {
  const { flashes, currentAirport } = useLiveData()

  const center: [number, number] = currentAirport
    ? [currentAirport.lat, currentAirport.lon]
    : [46.2276, 2.2137]

  return (
    <div className="col-span-6 relative w-full h-[600px] rounded-xl overflow-hidden border border-border bg-card shadow-sm">
      <Map center={center} zoom={11} className="z-0 h-full w-full">
        <MapLayers defaultLayerGroups={["Zones de sécurité", "Impacts récents"]}>
          <MapTileLayer name="Standard" />

          <MapLayerGroup name="Zones de sécurité">
            <MapCircle center={center} radius={20000} className="stroke-[var(--chart-2)] fill-[var(--chart-2)]" />
            <MapCircle center={center} radius={50000} className="stroke-muted-foreground fill-transparent stroke-1" />
            <MapMarker position={center}>
              <MapTooltip side="top">{currentAirport?.name ?? "Aéroport"}</MapTooltip>
            </MapMarker>
          </MapLayerGroup>

          <MapLayerGroup name="Impacts récents">
            <MapMarkerClusterGroup>
              {flashes.map((flash) => (
                <MapMarker
                  key={flash.rank}
                  position={[flash.lat, flash.lon]}
                  icon={
                    <Zap
                      className="size-6"
                      style={{
                        color: flash.flash_type === "CG" ? "var(--chart-2)" : "var(--chart-1)",
                        fill: flash.flash_type === "CG" ? "var(--chart-2)" : "var(--chart-1)",
                        filter: "drop-shadow(0 0 4px rgba(0,0,0,0.5))",
                      }}
                    />
                  }
                >
                  <MapPopup>
                    <div className="flex flex-col gap-1 text-center">
                      <p className="font-bold text-sm">Impact {flash.flash_type === "CG" ? "Nuage-Sol" : "Intra-Nuage"}</p>
                      <p className="text-xs text-muted-foreground">{new Date(flash.date).toLocaleTimeString("fr-FR", { timeZone: "UTC" })} UTC</p>
                      <p className="text-xs text-muted-foreground">{flash.lat.toFixed(4)}, {flash.lon.toFixed(4)}</p>
                      <p className="text-xs text-muted-foreground">Distance : {flash.dist_km.toFixed(1)} km</p>
                      <p className="text-xs font-semibold">Score : {(flash.score * 100).toFixed(0)}%</p>
                    </div>
                  </MapPopup>
                </MapMarker>
              ))}
            </MapMarkerClusterGroup>
          </MapLayerGroup>

          <MapLayersControl position="top-1 right-1" className="mt-2 mr-2" />
          <MapFullscreenControl position="top-1 left-1" className="mt-2 ml-2" />
          <MapZoomControl position="bottom-1 right-1" className="mb-2 mr-2" />
        </MapLayers>
      </Map>

      <Card className="absolute bottom-4 left-4 z-[1000] w-fit bg-card/80 backdrop-blur-md border-border shadow-md pointer-events-none">
        <CardContent className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="size-3 rounded-full border-2 border-[var(--chart-2)] bg-[var(--chart-2)]/20" />
            <span className="text-[10px] font-medium uppercase tracking-wider">Alerte (20km)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-3 border-t-2 border-dashed border-muted-foreground w-4" />
            <span className="text-[10px] font-medium uppercase tracking-wider">Surveillance (50km)</span>
          </div>
          <Separator className="bg-border" />
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5">
              <Zap className="size-3 text-[var(--chart-2)] fill-[var(--chart-2)]" />
              <span className="text-[10px] font-bold">CG</span>
            </div>
            <div className="flex items-center gap-1.5">
              <Zap className="size-3 text-[var(--chart-1)] fill-[var(--chart-1)]" />
              <span className="text-[10px] font-bold">IC</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
