import {
  createContext,
  useEffect,
  useReducer,
  useRef,
  useCallback,
  type ReactNode,
} from "react"
import { toast } from "sonner"
import { WSClient, type Flash, type Prediction, type AlertMeta } from "@/services/ws"
import { fetchAirports, fetchAlerts, fetchStats, type Airport, type AlertSummary, type AirportStats } from "@/services/api"
import { logger } from "@/utils/logger"

// ── Types ────────────────────────────────────────────────────────────────────

interface LiveDataState {
  airports: Airport[]
  selectedAirport: string
  isConnected: boolean
  isReplaying: boolean
  alertMeta: AlertMeta | null
  flashes: Flash[]
  currentFlash: Flash | null
  prediction: Prediction | null
  alertEnded: boolean
  speed: number
  stats: AirportStats | null
  alertsHistory: AlertSummary[]
  error: string | null
}

interface LiveDataContextType extends LiveDataState {
  selectAirport: (airport: string) => void
  setSpeed: (speed: number) => void
  currentAirport: Airport | null
  refreshStats: () => void
  clearError: () => void
}

// ── Reducer ──────────────────────────────────────────────────────────────────

type Action =
  | { type: "SET_AIRPORTS"; airports: Airport[] }
  | { type: "SELECT_AIRPORT"; airport: string }
  | { type: "SET_CONNECTED"; connected: boolean }
  | { type: "SUBSCRIBED"; meta: AlertMeta }
  | { type: "FLASH"; flash: Flash }
  | { type: "PREDICTION"; prediction: Prediction }
  | { type: "ALERT_END" }
  | { type: "SET_SPEED"; speed: number }
  | { type: "SET_STATS"; stats: AirportStats }
  | { type: "SET_HISTORY"; history: AlertSummary[] }
  | { type: "SET_ERROR"; error: string | null }

const initialState: LiveDataState = {
  airports: [],
  selectedAirport: "bastia",
  isConnected: false,
  isReplaying: false,
  alertMeta: null,
  flashes: [],
  currentFlash: null,
  prediction: null,
  alertEnded: false,
  speed: 10,
  stats: null,
  alertsHistory: [],
  error: null,
}

function reducer(state: LiveDataState, action: Action): LiveDataState {
  switch (action.type) {
    case "SET_AIRPORTS":
      return { ...state, airports: action.airports }
    case "SELECT_AIRPORT":
      return {
        ...state,
        selectedAirport: action.airport,
        isReplaying: false,
        alertMeta: null,
        flashes: [],
        currentFlash: null,
        prediction: null,
        alertEnded: false,
        stats: null,
        alertsHistory: [],
        error: null,
      }
    case "SET_CONNECTED":
      return { ...state, isConnected: action.connected }
    case "SUBSCRIBED":
      return {
        ...state,
        isReplaying: true,
        alertMeta: action.meta,
        flashes: [],
        currentFlash: null,
        prediction: null,
        alertEnded: false,
        error: null,
      }
    case "FLASH":
      return {
        ...state,
        flashes: [...state.flashes, action.flash],
        currentFlash: action.flash,
      }
    case "PREDICTION":
      return { ...state, prediction: action.prediction }
    case "ALERT_END":
      return { ...state, isReplaying: false, alertEnded: true }
    case "SET_SPEED":
      return { ...state, speed: action.speed }
    case "SET_STATS":
      return { ...state, stats: action.stats }
    case "SET_HISTORY":
      return { ...state, alertsHistory: action.history }
    case "SET_ERROR":
      return { ...state, error: action.error }
    default:
      return state
  }
}

// ── Context ──────────────────────────────────────────────────────────────────

const LiveDataContext = createContext<LiveDataContextType | null>(null)

export function LiveDataProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState)
  const wsRef = useRef<WSClient | null>(null)
  const speedRef = useRef(initialState.speed)
  const lastErrorRef = useRef<string | null>(null)

  const handleError = useCallback((error: unknown, context: string) => {
    const message = error instanceof Error ? error.message : String(error)
    logger.error(`${context}:`, message)
    
    // Only show toast if it's a new error (not a duplicate)
    const errorKey = `${context}:${message}`
    if (lastErrorRef.current !== errorKey) {
      lastErrorRef.current = errorKey
      toast.error(context, {
        description: message,
        duration: 5000,
      })
    }
    
    dispatch({ type: "SET_ERROR", error: `${context}: ${message}` })
  }, [])

  const clearError = useCallback(() => {
    dispatch({ type: "SET_ERROR", error: null })
    lastErrorRef.current = null
  }, [])

  // Load airports on mount
  useEffect(() => {
    console.log("Fetching airports...")
    fetchAirports()
      .then((airports) => {
        console.log("Airports loaded:", airports.length)
        dispatch({ type: "SET_AIRPORTS", airports })
        clearError()
      })
      .catch((error) => {
        console.error("Failed to load airports:", error)
        handleError(error, "Failed to load airports")
      })
  }, [handleError, clearError])

  // Init WS client
  useEffect(() => {
    console.log("WS: Initializing WebSocket client...")
    const client = new WSClient(
      (msg) => {
        console.log("WS: Message received:", msg.type, msg)
        if (msg.type === "subscribed") {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { type: _, ...meta } = msg
          console.log("WS: Subscribed with meta:", meta)
          dispatch({ type: "SUBSCRIBED", meta })
        } else if (msg.type === "flash") {
          console.log("WS: Flash received, rank:", msg.data.rank)
          dispatch({ type: "FLASH", flash: msg.data })
        } else if (msg.type === "prediction_triggered") {
          console.log("WS: Prediction triggered:", msg.data)
          dispatch({ type: "PREDICTION", prediction: msg.data })
        } else if (msg.type === "alert_end") {
          console.log("WS: Alert end")
          dispatch({ type: "ALERT_END" })
        } else if (msg.type === "error") {
          handleError(msg.message, "WebSocket error")
        }
      },
      (connected) => {
        console.log("WS: Connection status changed:", connected)
        dispatch({ type: "SET_CONNECTED", connected })
        if (connected) {
          toast.success("Connecté au serveur")
        } else {
          toast.error("Déconnecté du serveur", {
            description: "Tentative de reconnection...",
            duration: 3000,
          })
        }
      }
    )

    wsRef.current = client
    console.log("WS: Connecting...")
    client.connect()

    return () => {
      console.log("WS: Cleanup")
      client.disconnect()
    }
  }, [handleError])

  // Subscribe when airport changes
  useEffect(() => {
    if (!wsRef.current) {
      console.log("WS: Not ready yet, waiting...")
      return
    }
    console.log("WS: Subscribing to airport:", state.selectedAirport)
    wsRef.current?.subscribe(state.selectedAirport)
    wsRef.current?.setSpeed(speedRef.current)
  }, [state.selectedAirport])

  // Fetch stats and history when airport changes
  useEffect(() => {
    const ap = state.selectedAirport
    if (!ap) return

    fetchStats(ap)
      .then((stats) => {
        dispatch({ type: "SET_STATS", stats })
        clearError()
      })
      .catch((error) => handleError(error, "Failed to load stats"))

    fetchAlerts(ap)
      .then((history) => {
        dispatch({ type: "SET_HISTORY", history })
        clearError()
      })
      .catch((error) => handleError(error, "Failed to load alerts"))
  }, [state.selectedAirport, handleError, clearError])

  // Refresh stats when an alert ends
  useEffect(() => {
    if (state.alertEnded) {
      const ap = state.selectedAirport
      fetchStats(ap)
        .then((stats) => {
          dispatch({ type: "SET_STATS", stats })
          clearError()
        })
        .catch((error) => handleError(error, "Failed to refresh stats"))

      fetchAlerts(ap)
        .then((history) => {
          dispatch({ type: "SET_HISTORY", history })
          clearError()
        })
        .catch((error) => handleError(error, "Failed to refresh alerts"))
    }
  }, [state.alertEnded, state.selectedAirport, handleError, clearError])

  const selectAirport = useCallback((airport: string) => {
    dispatch({ type: "SELECT_AIRPORT", airport })
  }, [])

  const setSpeed = useCallback((speed: number) => {
    speedRef.current = speed
    dispatch({ type: "SET_SPEED", speed })
    wsRef.current?.setSpeed(speed)
  }, [])

  const refreshStats = useCallback(() => {
    fetchStats(state.selectedAirport)
      .then((stats) => {
        dispatch({ type: "SET_STATS", stats })
        clearError()
      })
      .catch((error) => handleError(error, "Failed to refresh stats"))
  }, [state.selectedAirport, handleError, clearError])

  const currentAirport =
    state.airports.find((a) => a.id === state.selectedAirport) ?? null

  return (
    <LiveDataContext.Provider
      value={{ ...state, selectAirport, setSpeed, currentAirport, refreshStats, clearError }}
    >
      {children}
    </LiveDataContext.Provider>
  )
}

export default LiveDataContext
