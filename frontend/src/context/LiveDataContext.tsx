import {
  createContext,
  useContext,
  useEffect,
  useReducer,
  useRef,
  useCallback,
  type ReactNode,
} from "react"
import { WSClient, type Flash, type Prediction, type AlertMeta } from "@/services/ws"
import { fetchAirports, type Airport } from "@/services/api"

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
}

interface LiveDataContextType extends LiveDataState {
  selectAirport: (airport: string) => void
  setSpeed: (speed: number) => void
  currentAirport: Airport | null
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

  // Subscribe when airport changes, and sync current speed to server
  useEffect(() => {
    wsRef.current?.subscribe(state.selectedAirport)
    wsRef.current?.setSpeed(speedRef.current)
  }, [state.selectedAirport])

  const selectAirport = useCallback((airport: string) => {
    dispatch({ type: "SELECT_AIRPORT", airport })
  }, [])

  const setSpeed = useCallback((speed: number) => {
    speedRef.current = speed
    dispatch({ type: "SET_SPEED", speed })
    wsRef.current?.setSpeed(speed)
  }, [])

  const currentAirport =
    state.airports.find((a) => a.id === state.selectedAirport) ?? null

  return (
    <LiveDataContext.Provider value={{ ...state, selectAirport, setSpeed, currentAirport }}>
      {children}
    </LiveDataContext.Provider>
  )
}

export function useLiveData() {
  const ctx = useContext(LiveDataContext)
  if (!ctx) throw new Error("useLiveData must be used inside LiveDataProvider")
  return ctx
}
