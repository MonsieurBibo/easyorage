const BASE = "http://localhost:8000"

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

export async function fetchAirports(): Promise<Airport[]> {
  const res = await fetch(`${BASE}/airports`)
  if (!res.ok) throw new Error("Failed to fetch airports")
  return res.json()
}

export async function fetchAlerts(airport: string): Promise<AlertSummary[]> {
  const res = await fetch(`${BASE}/airports/${airport}/alerts`)
  if (!res.ok) throw new Error(`Failed to fetch alerts for ${airport}`)
  return res.json()
}

export async function fetchStats(airport: string): Promise<AirportStats> {
  const res = await fetch(`${BASE}/airports/${airport}/stats`)
  if (!res.ok) throw new Error(`Failed to fetch stats for ${airport}`)
  return res.json()
}
