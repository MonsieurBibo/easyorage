const WS_URL = "ws://localhost:8000/ws"

export interface Flash {
  rank: number
  date: string
  lat: number
  lon: number
  flash_type: "CG" | "IC"
  dist_km: number
  amplitude: number
  score: number
  prediction_triggered: boolean
}

export interface Prediction {
  triggered_at_rank: number
  triggered_at_date: string
  confidence: number
}

export interface AlertMeta {
  alert_id: string
  airport: string
  n_flashes: number
  duration_s: number
  start_date: string
  end_date: string
}

type WSMessage =
  | ({ type: "subscribed" } & AlertMeta)
  | { type: "flash"; data: Flash }
  | { type: "prediction_triggered"; data: Prediction }
  | { type: "alert_end"; data: { n_flashes: number; end_date: string; prediction: Prediction | null } }
  | { type: "error"; message: string }

type MessageHandler = (msg: WSMessage) => void
type StatusHandler = (connected: boolean) => void

export class WSClient {
  private ws: WebSocket | null = null
  private onMessage: MessageHandler
  private onStatus: StatusHandler
  private pendingAirport: string | null = null

  constructor(onMessage: MessageHandler, onStatus: StatusHandler) {
    this.onMessage = onMessage
    this.onStatus = onStatus
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return
    this.ws = new WebSocket(WS_URL)

    this.ws.onopen = () => {
      this.onStatus(true)
      if (this.pendingAirport) {
        this._send({ action: "subscribe", airport: this.pendingAirport })
        this.pendingAirport = null
      }
    }

    this.ws.onmessage = (e) => {
      try {
        const msg: WSMessage = JSON.parse(e.data)
        this.onMessage(msg)
      } catch {
        console.error("WS parse error", e.data)
      }
    }

    this.ws.onclose = () => {
      this.onStatus(false)
    }

    this.ws.onerror = () => {
      this.onStatus(false)
    }
  }

  subscribe(airport: string, alertId?: string) {
    const payload: Record<string, unknown> = { action: "subscribe", airport }
    if (alertId) payload.alert_id = alertId
    if (this.ws?.readyState === WebSocket.OPEN) {
      this._send(payload)
    } else {
      this.pendingAirport = airport
      this.connect()
    }
  }

  setSpeed(speed: number) {
    this._send({ action: "set_speed", speed })
  }

  disconnect() {
    this.ws?.close()
    this.ws = null
  }

  private _send(data: object) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    }
  }
}
