import { createRootRoute, createRoute, createRouter } from '@tanstack/react-router'
import { DashboardLayout } from './components/DashboardLayout'
import { Dashboard } from './components/Dashboard'
import { Analytics } from './components/dashboard/analytics'
import { Reports } from './components/dashboard/reports'

const rootRoute = createRootRoute({
  component: DashboardLayout,
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: Dashboard,
})

const analyticsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/analytics',
  component: Analytics,
})

const reportsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/reports',
  component: Reports,
})

const routeTree = rootRoute.addChildren([indexRoute, analyticsRoute, reportsRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
