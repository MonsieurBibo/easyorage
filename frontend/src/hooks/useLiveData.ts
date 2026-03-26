import { useContext } from "react"
import LiveDataContext from "@/context/LiveDataContext"

export function useLiveData() {
  const ctx = useContext(LiveDataContext)
  if (!ctx) throw new Error("useLiveData must be used inside LiveDataProvider")
  return ctx
}
