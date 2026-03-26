import { z } from "zod"
import { logger } from "@/utils/logger"

const WS_URL = import.meta.env.VITE_WS_BASE_URL || "ws://localhost:8000/ws"
const RECONNECT_DELAY = 1000 // 1 second
const MAX_RECONNECT_DELAY = 30000 // 30 seconds
const MAX_RECONNECT_ATTEMPTS = 5

// Zod schemas for WebSocket messages
const FlashSchema = z.object({
  rank: z.number(),
  date: z.string(),
  lat: z.number(),
  lon: z.number(),
  flash_type: z.enum(["CG", "IC"]),
  dist_km: z.number(),
  amplitude: z.number(),
  score: z.number(),
  prediction_triggered: z.boolean(),
})

const PredictionSchema = z.object({
  triggered_at_rank: z.number(),
  triggered_at_date: z.string(),
  confidence: z.number(),
})

const AlertMetaSchema = z.object({
  alert_id: z.string(),
  airport: z.string(),
  n_flashes: z.number(),
  duration_s: z.number(),
  start_date: z.string(),
  end_date: z.string(),
})

const SubscribedMessageSchema = z.object({
  type: z.literal("subscribed"),
}).merge(AlertMetaSchema)

const FlashMessageSchema = z.object({
  type: z.literal("flash"),
  data: FlashSchema,
})

const PredictionMessageSchema = z.object({
  type: z.literal("prediction_triggered"),
  data: PredictionSchema,
})

const AlertEndMessageSchema = z.object({
  type: z.literal("alert_end"),
  data: z.object({
    n_flashes: z.number(),
    end_date: z.string(),
    prediction: PredictionSchema.nullable(),
  }),
})

const ErrorMessageSchema = z.object({
  type: z.literal("error"),
  message: z.string(),
})

const WSMessageSchema = z.discriminatedUnion("type", [
  SubscribedMessageSchema,
  FlashMessageSchema,
  PredictionMessageSchema,
  AlertEndMessageSchema,
  ErrorMessageSchema,
])

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

export type WSMessage =
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
  private pendingAlertId: string | null = null
  private reconnectAttempts = 0
  private reconnectDelay = RECONNECT_DELAY
  private shouldReconnect = true

  constructor(onMessage: MessageHandler, onStatus: StatusHandler) {
    this.onMessage = onMessage
    this.onStatus = onStatus
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return

    try {
      this.ws = new WebSocket(WS_URL)

      this.ws.onopen = () => {
        logger.info("WebSocket connected")
        this.reconnectAttempts = 0
        this.reconnectDelay = RECONNECT_DELAY
        this.onStatus(true)
        if (this.pendingAirport) {
          this._send({ action: "subscribe", airport: this.pendingAirport, alert_id: this.pendingAlertId || undefined })
          this.pendingAirport = null
          this.pendingAlertId = null
        }
      }

      this.ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data)
          const parsed = WSMessageSchema.parse(data)
          this.onMessage(parsed)
        } catch (error) {
          if (error instanceof z.ZodError) {
            logger.error("WS message validation error:", error.issues, "data:", e.data)
          } else {
            logger.error("WS parse error:", e.data)
          }
        }
      }

      this.ws.onclose = () => {
        logger.info("WebSocket disconnected")
        this.onStatus(false)
        this._scheduleReconnect()
      }

      this.ws.onerror = (error) => {
        logger.error("WebSocket error:", error)
        this.onStatus(false)
      }
    } catch (error) {
      logger.error("Failed to create WebSocket:", error)
      this.onStatus(false)
      this._scheduleReconnect()
    }
  }

  private _scheduleReconnect() {
    if (!this.shouldReconnect || this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      logger.error("Max reconnection attempts reached")
      this.shouldReconnect = true // Reset for next manual connect
      return
    }

    this.reconnectAttempts++
    logger.info(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`)

    setTimeout(() => {
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, MAX_RECONNECT_DELAY) // Exponential backoff
      this.connect()
    }, this.reconnectDelay)
  }

  subscribe(airport: string, alertId?: string) {
    const payload: Record<string, unknown> = { action: "subscribe", airport }
    if (alertId) payload.alert_id = alertId

    if (this.ws?.readyState === WebSocket.OPEN) {
      this._send(payload)
    } else {
      this.pendingAirport = airport
      this.pendingAlertId = alertId || null
      this.shouldReconnect = true
      this.connect()
    }
  }

  setSpeed(speed: number) {
    this._send({ action: "set_speed", speed })
  }

  disconnect() {
    this.shouldReconnect = false
    this.ws?.close()
    this.ws = null
    this.pendingAirport = null
    this.pendingAlertId = null
    this.reconnectAttempts = 0
  }

  private _send(data: object) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      logger.warn("Attempted to send message while WebSocket is not open")
    }
  }
}
