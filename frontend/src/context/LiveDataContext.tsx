import {
  createContext,
  useEffect,
  useReducer,
  useRef,
  useCallback,
  type ReactNode,
} from "react"
import { WSClient, type Flash, type Prediction, type AlertMeta } from "@/services/ws"
import { fetchAirports, fetchAlerts, fetchStats, type Airport, type AlertSummary, type AirportStats } from "@/services/api"

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
}

interface LiveDataContextType extends LiveDataState {
  selectAirport: (airport: string) => void
  setSpeed: (speed: number) => void
  currentAirport: Airport | null
  refreshStats: () => void
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

  // Load airports on mount
  useEffect(() => {
    fetchAirports()
      .then((airports) => dispatch({ type: "SET_AIRPORTS", airports }))
      .catch(console.error)
  }, [])

  // Init WS client
  useEffect(() => {
    const client = new WSClient(
      (msg) => {
        if (msg.type === "subscribed") {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { type: _, ...meta } = msg
          dispatch({ type: "SUBSCRIBED", meta: meta as AlertMeta })
        } else if (msg.type === "flash") {
          dispatch({ type: "FLASH", flash: msg.data })
        } else if (msg.type === "prediction_triggered") {
          dispatch({ type: "PREDICTION", prediction: msg.data })
        } else if (msg.type === "alert_end") {
          dispatch({ type: "ALERT_END" })
        } else if (msg.type === "error") {
          console.error("WS error:", msg.message)
        }
      },
      (connected) => dispatch({ type: "SET_CONNECTED", connected })
    )

    wsRef.current = client
    client.connect()

    return () => client.disconnect()
  }, [])

  // Fetch stats and history when airport changes
  useEffect(() => {
    const ap = state.selectedAirport
    if (!ap) return

    fetchStats(ap)
      .then((stats) => dispatch({ type: "SET_STATS", stats }))
      .catch(console.error)

    fetchAlerts(ap)
      .then((history) => dispatch({ type: "SET_HISTORY", history }))
      .catch(console.error)
  }, [state.selectedAirport])

  // Subscribe when airport changes, and sync current speed to server
  useEffect(() => {
    wsRef.current?.subscribe(state.selectedAirport)
    wsRef.current?.setSpeed(speedRef.current)
  }, [state.selectedAirport])

  // Refresh stats when an alert ends
  useEffect(() => {
    if (state.alertEnded) {
      const ap = state.selectedAirport
      fetchStats(ap)
        .then((stats) => dispatch({ type: "SET_STATS", stats }))
        .catch(console.error)

      fetchAlerts(ap)
        .then((history) => dispatch({ type: "SET_HISTORY", history }))
        .catch(console.error)
    }
  }, [state.alertEnded, state.selectedAirport])

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
      .then((stats) => dispatch({ type: "SET_STATS", stats }))
      .catch(console.error)
  }, [state.selectedAirport])

  const currentAirport =
    state.airports.find((a) => a.id === state.selectedAirport) ?? null

  return (
    <LiveDataContext.Provider
      value={{ ...state, selectAirport, setSpeed, currentAirport, refreshStats }}
    >
      {children}
    </LiveDataContext.Provider>
  )
}

export default LiveDataContext