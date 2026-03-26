import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { LucideIcon } from "lucide-react"
import type { ChartConfig } from "@/components/ui/chart"
import { MiniChart } from "@/components/dashboard/chart-line"

interface StatCardProps {
  title: string
  value: string
  description: string
  icon: LucideIcon
  trendColor?: string
  chartData?: Array<{ [key: string]: string | number }>
  chartConfig?: ChartConfig
  chartDataKey?: string
}

export function StatCard({
  title,
  value,
  description,
  icon: Icon,
  trendColor,
  chartData,
  chartConfig,
  chartDataKey,
}: StatCardProps) {
  return (
    <Card className="border-border bg-card shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-foreground">
          {title}
        </CardTitle>
        <Icon className="h-4 w-4" style={{ color: trendColor }} />
      </CardHeader>
      <CardContent className="flex items-end justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-2xl font-bold text-foreground">{value}</div>
          <p className="text-xs text-muted-foreground truncate">
            {description}
          </p>
        </div>
        {chartData && chartConfig && chartDataKey && (
          <MiniChart
            chartData={chartData}
            chartConfig={chartConfig}
            dataKey={chartDataKey}
            trendColor={trendColor}
          />
        )}
      </CardContent>
    </Card>
  )
}
