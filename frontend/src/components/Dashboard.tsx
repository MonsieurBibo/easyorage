import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Zap,
  Clock,
  AlertTriangle,
  PlaneTakeoff,
} from "lucide-react"

import { DashboardHeader } from "./dashboard/header"
import { StatCard } from "./dashboard/stat-card"
import { LiveMap } from "./dashboard/live-map"
import { RecentAlerts } from "./dashboard/recent-alerts"
import { DatePickerInput } from "./dashboard/date-picker"

export function Dashboard() {
  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      <DashboardHeader />

      <main className="flex-1 space-y-4 p-8 pt-6">
        <div className="flex items-center justify-between space-y-2">
          <h2 className="text-3xl font-bold tracking-tight text-foreground">Dashboard - Bastia</h2>
          <div className="flex items-center space-x-2">
            <DatePickerInput/>
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90 ml-2">
              Exporter rapport
            </Button>
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
                value="89%"
                description="Confiance élevée (Model v2.1)"
                icon={Zap}
                trendColor="var(--chart-1)"
              />
              <StatCard 
                title="Temps restant"
                value="12min"
                description="Avant reprise d'activité"
                icon={Clock}
                trendColor="var(--chart-2)"
              />
              <StatCard 
                title="Impacts (20km)"
                value="14"
                description="Derniers 30 minutes"
                icon={AlertTriangle}
                trendColor="var(--chart-3)"
              />
              <StatCard 
                title="État de l'aéroport"
                value="EN ALERTE"
                description="Alerte active depuis 42min"
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
  