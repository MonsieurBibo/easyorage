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
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-0.5 py-1 px-2">
        <CardTitle className="text-[10px] font-medium text-foreground leading-none">
          {title}
        </CardTitle>
        <Icon className="h-2.5 w-2.5" style={{ color: trendColor }} />
      </CardHeader>
      <CardContent className="flex items-start justify-between gap-1.5 px-2 pb-1">
        <div className="flex-1 min-w-0">
          <div className="text-sm font-bold text-foreground leading-none">{value}</div>
          <p className="text-[9px] text-muted-foreground truncate mt-0.5">
            {description}
          </p>
        </div>
        {chartData && chartConfig && chartDataKey && (
          <div className="h-8 w-26 mr-10 space-y-10">
            <MiniChart
              chartData={chartData}
              chartConfig={chartConfig}
              dataKey={chartDataKey}
              trendColor={trendColor}
            />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
