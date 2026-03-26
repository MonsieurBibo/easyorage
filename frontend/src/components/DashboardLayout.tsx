import { Outlet } from "@tanstack/react-router"
import { DashboardHeader } from "./dashboard/header"

export function DashboardLayout() {
  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground">
      <DashboardHeader />
      <main className="flex-1 p-8 pt-6">
        <Outlet />
      </main>
    </div>
  )
}
