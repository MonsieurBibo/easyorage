import { z } from "zod"

const BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
const DEFAULT_TIMEOUT = 10000 // 10 seconds

// Zod schemas for runtime validation
const AirportSchema = z.object({
  id: z.string(),
  name: z.string(),
  lat: z.number(),
  lon: z.number(),
})

const PredictionSchema = z.object({
  triggered_at_rank: z.number(),
  triggered_at_date: z.string(),
  confidence: z.number(),
})

const AlertSummarySchema = z.object({
  alert_id: z.string(),
  airport: z.string(),
  start_date: z.string(),
  end_date: z.string(),
  n_flashes: z.number(),
  duration_s: z.number(),
  prediction: PredictionSchema.nullable(),
})

const AirportStatsSchema = z.object({
  airport: z.string(),
  total_alerts: z.number(),
  covered_alerts: z.number(),
  coverage_rate: z.number(),
  total_gain_h: z.number(),
  risk: z.number(),
})

export interface Airport {
  id: string
  name: string
  lat: number
  lon: number
}

export interface AlertSummary {
  alert_id: string
  airport: string
  start_date: string
  end_date: string
  n_flashes: number
  duration_s: number
  prediction: { triggered_at_rank: number; triggered_at_date: string; confidence: number } | null
}

export interface AirportStats {
  airport: string
  total_alerts: number
  covered_alerts: number
  coverage_rate: number
  total_gain_h: number
  risk: number
}

async function fetchWithTimeout<T>(
  url: string,
  schema: z.ZodSchema<T>,
  timeout = DEFAULT_TIMEOUT
): Promise<T> {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeout)

  try {
    const res = await fetch(url, { signal: controller.signal })

    if (!res.ok) {
      let errorMessage = `HTTP ${res.status}: ${res.statusText}`
      try {
        const errorData = await res.json()
        errorMessage = errorData.detail || errorData.message || errorMessage
      } catch {
        // Ignore parse error, use default message
      }
      throw new Error(errorMessage)
    }

    const data = await res.json()
    return schema.parse(data)
  } finally {
    clearTimeout(id)
  }
}

export async function fetchAirports(): Promise<Airport[]> {
  const res = await fetchWithTimeout(`${BASE}/airports`, z.array(AirportSchema))
  return res
}

export async function fetchAlerts(airport: string): Promise<AlertSummary[]> {
  const res = await fetchWithTimeout(
    `${BASE}/airports/${airport}/alerts`,
    z.array(AlertSummarySchema)
  )
  return res
}

export async function fetchStats(airport: string): Promise<AirportStats> {
  const res = await fetchWithTimeout(
    `${BASE}/airports/${airport}/stats`,
    AirportStatsSchema
  )
  return res
}
