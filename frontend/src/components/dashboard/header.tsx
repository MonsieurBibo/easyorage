import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { ChevronDown, Menu } from "lucide-react"
import { ThemeSwitcher } from "@/components/ui/mode-toogle"
import { Link } from "@tanstack/react-router"
import { useLiveData } from "@/hooks/useLiveData"
import { useState, useRef, useEffect } from "react"

export function DashboardHeader() {
  const { airports, selectedAirport, selectAirport, isConnected } = useLiveData()
  const [open, setOpen] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const mobileMenuRef = useRef<HTMLDivElement>(null)

  const current = airports.find((a) => a.id === selectedAirport)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
      if (mobileMenuRef.current && !mobileMenuRef.current.contains(e.target as Node)) setMobileMenuOpen(false)
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [])

  return (
    <header className="border-b border-border bg-card">
      <div className="flex h-12 items-center px-3 md:px-4 gap-4">

        {/* Airport selector */}
        <div className="relative" ref={ref}>
          <button
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-1.5 px-2.5 py-1 border border-border rounded-md shadow-sm cursor-pointer hover:bg-muted transition-colors"
          >
            <span className="text-xs font-medium text-foreground">
              {current?.name ?? "Aéroport"}
            </span>
            <ChevronDown className="h-2.5 w-2.5 text-muted-foreground" />
          </button>

          {open && airports.length > 0 && (
            <div className="absolute top-full left-0 mt-1 w-40 bg-card border border-border rounded-md shadow-lg z-50">
              {airports.map((ap) => (
                <button
                  key={ap.id}
                  onClick={() => { selectAirport(ap.id); setOpen(false) }}
                  className={`w-full text-left px-3 py-1.5 text-sm hover:bg-muted transition-colors ${ap.id === selectedAirport ? "font-semibold text-foreground" : "text-muted-foreground"}`}
                >
                  {ap.name}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-3 text-xs font-medium">
          <Link to="/" className="text-foreground hover:text-primary [&.active]:font-bold">Dashboard</Link>
          <Link to="/analytics" className="text-muted-foreground hover:text-primary [&.active]:font-bold [&.active]:text-foreground">Statistiques</Link>
          <Link to="/reports" className="text-muted-foreground hover:text-primary [&.active]:font-bold [&.active]:text-foreground">Rapports d'alerte</Link>
        </nav>

        {/* Mobile menu button */}
        <button
          onClick={() => setMobileMenuOpen((o) => !o)}
          className="md:hidden p-1.5 rounded-md hover:bg-muted transition-colors"
        >
          <Menu className="h-4 w-4 text-foreground" />
        </button>

        <div className="ml-auto flex items-center gap-2">
          <div className={`h-1.5 w-1.5 rounded-full ${isConnected ? "bg-green-500" : "bg-red-400"}`} title={isConnected ? "Connecté" : "Déconnecté"} />
          <Avatar className="h-6 w-6">
            <AvatarFallback className="text-xs">U</AvatarFallback>
          </Avatar>
          <ThemeSwitcher />
        </div>
      </div>

      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div ref={mobileMenuRef} className="md:hidden border-t border-border bg-card px-3 py-2 space-y-1">
          <Link
            to="/"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 text-sm font-medium rounded-md hover:bg-muted [&.active]:font-bold [&.active]:bg-muted"
          >
            Dashboard
          </Link>
          <Link
            to="/analytics"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 text-sm font-medium rounded-md hover:bg-muted [&.active]:font-bold [&.active]:bg-muted"
          >
            Statistiques
          </Link>
          <Link
            to="/reports"
            onClick={() => setMobileMenuOpen(false)}
            className="block px-3 py-2 text-sm font-medium rounded-md hover:bg-muted [&.active]:font-bold [&.active]:bg-muted"
          >
            Rapports d'alerte
          </Link>
        </div>
      )}
    </header>
  )
}
