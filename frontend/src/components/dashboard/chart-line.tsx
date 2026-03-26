import { CartesianGrid, Line, LineChart, XAxis } from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"

export const description = "A line chart"

interface ChartLineDefaultProps {
  chartData: Array<{ [key: string]: string | number }>
  chartConfig: ChartConfig
  dataKey: string
  title?: string
  description?: string
  footer?: string
  className?: string
}

export function ChartLineDefault({
  chartData,
  chartConfig,
  dataKey,
  title = "Line Chart",
  description,
  footer,
  className,
}: ChartLineDefaultProps) {
  const firstKey = Object.keys(chartConfig)[0]
  const colorVar = firstKey ? `var(--color-${firstKey})` : "var(--chart-1)"

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <LineChart
            accessibilityLayer
            data={chartData}
            margin={{
              left: 12,
              right: 12,
            }}
          >
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => value.slice(0, 3)}
            />
            <ChartTooltip
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Line
              dataKey={dataKey}
              type="natural"
              stroke={colorVar}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
      {footer && (
        <CardFooter className="flex-col items-start gap-2 text-sm">
          <div className="leading-none text-muted-foreground">{footer}</div>
        </CardFooter>
      )}
    </Card>
  )
}

// Mini chart variant for StatCards
interface MiniChartProps {
  chartData: Array<{ [key: string]: string | number }>
  chartConfig: ChartConfig
  dataKey: string
  trendColor?: string
}

export function MiniChart({ chartData, chartConfig, dataKey, trendColor }: MiniChartProps) {
  const firstKey = Object.keys(chartConfig)[0]
  const colorVar = trendColor || (firstKey ? `var(--color-${firstKey})` : "var(--chart-1)")

  return (
    <ChartContainer config={chartConfig} className="h-16 w-32 mt-2">
      <LineChart
        accessibilityLayer
        data={chartData}
        margin={{ top: 4, right: 4, bottom: 4, left: 4 }}
      >
        <CartesianGrid vertical={true} horizontal={true} strokeDasharray="2 2" stroke="var(--muted-foreground)" opacity={0.15} />
        <XAxis
          dataKey="date"
          tick={false}
          axisLine={false}
        />
        <Line
          dataKey={dataKey}
          type="monotone"
          stroke={colorVar}
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ChartContainer>
  )
}
