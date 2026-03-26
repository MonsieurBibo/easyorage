import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { ChevronDown } from "lucide-react"
import { ThemeSwitcher } from "@/components/ui/mode-toogle"
import { Link } from "@tanstack/react-router"
import { useLiveData } from "@/hooks/useLiveData"
import { useState, useRef, useEffect } from "react"

export function DashboardHeader() {
  const { airports, selectedAirport, selectAirport, isConnected } = useLiveData()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  const current = airports.find((a) => a.id === selectedAirport)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  return (
    <header className="border-b border-border bg-card">
      <div className="flex h-16 items-center px-4 md:px-8 gap-8">

        {/* Airport selector */}
        <div className="relative" ref={ref}>
          <button
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-2 px-3 py-1.5 border border-border rounded-md shadow-sm cursor-pointer hover:bg-muted transition-colors"
          >
            <span className="text-sm font-medium text-foreground">
              {current?.name ?? "Aéroport"}
            </span>
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </button>

          {open && airports.length > 0 && (
            <div className="absolute top-full left-0 mt-1 w-40 bg-card border border-border rounded-md shadow-lg z-50">
              {airports.map((ap) => (
                <button
                  key={ap.id}
                  onClick={() => { selectAirport(ap.id); setOpen(false) }}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-muted transition-colors ${ap.id === selectedAirport ? "font-semibold text-foreground" : "text-muted-foreground"}`}
                >
                  {ap.name}
                </button>
              ))}
            </div>
          )}
        </div>

        <nav className="flex items-center gap-6 text-sm font-medium">
          <Link to="/" className="text-foreground hover:text-primary [&.active]:font-bold">Dashboard</Link>
          <Link to="/analytics" className="text-muted-foreground hover:text-primary [&.active]:font-bold [&.active]:text-foreground">Statistiques</Link>
          <Link to="/reports" className="text-muted-foreground hover:text-primary [&.active]:font-bold [&.active]:text-foreground">Rapports d'alerte</Link>
        </nav>

        <div className="ml-auto flex items-center gap-4">
          <div className={`h-2 w-2 rounded-full ${isConnected ? "bg-green-500" : "bg-red-400"}`} title={isConnected ? "Connecté" : "Déconnecté"} />
          <Avatar className="h-8 w-8">
            <AvatarFallback>U</AvatarFallback>
          </Avatar>
          <ThemeSwitcher />
        </div>
      </div>
    </header>
  )
}
