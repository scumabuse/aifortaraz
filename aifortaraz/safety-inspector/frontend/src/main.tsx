import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  lazy,
  Suspense,
} from "react";
import ReactDOM from "react-dom/client";
import {
  BrowserRouter,
  Link,
  NavLink,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  Activity,
  ArrowDownLeft,
  ArrowRight,
  Camera,
  ChevronRight,
  CircleHelp,
  ClipboardList,
  FileText,
  LayoutDashboard,
  LogOut,
  Map,
  Menu,
  Monitor,
  Radio,
  Search,
  Settings,
  ShieldCheck,
  X,
} from "lucide-react";
import {
  api,
  clearSession,
  readSession,
  saveSession,
  type Session,
  type User,
} from "./api";
import { Button } from "./components/ui/button";
import ru from "./locales/ru.json";
import "./styles.css";
import { CamerasPage, CameraPage } from "./features/cameras/Cameras";
const ZoneEditor = lazy(() =>
  import("./features/zones/ZoneEditor").then((module) => ({
    default: module.ZoneEditor,
  })),
);
import { QueryClientProvider, useQuery } from "@tanstack/react-query";
import { queryClient } from "./lib/query";
import {
  RealtimeSession,
  ConnectionStatus,
  SoundButton,
  NotificationCenter,
} from "./features/realtime/Realtime";
import { IncidentsPage } from "./features/incidents/IncidentsPage";
import { IncidentPage } from "./features/incidents/IncidentPage";
import { DemoBadge } from "./features/live/DemoSettings";
const LivePage = lazy(() =>
  import("./features/live/LivePage").then((module) => ({
    default: module.LivePage,
  })),
);
const AboutPage = lazy(() => import("./features/project/AboutPage"));
const WallPage = lazy(() => import("./features/project/WallPage"));
const SettingsPage = lazy(() => import("./features/settings/SettingsPage"));
const ReportsPage = lazy(() => import("./features/reports/ReportsPage"));
const ActPage = lazy(() =>
  import("./features/reports/ReportsPage").then((module) => ({
    default: module.ActPage,
  })),
);
const DashboardPage = lazy(() => import("./features/analytics/DashboardPage"));
const AnalyticsPage = lazy(() => import("./features/analytics/AnalyticsPage"));
const PlanPage = lazy(() => import("./features/analytics/PlanPage"));

const navigation = [
  { path: "/", key: "dashboard", icon: LayoutDashboard },
  { path: "/live", key: "live", icon: Radio },
  { path: "/cameras", key: "cameras", icon: Camera },
  { path: "/incidents", key: "incidents", icon: ClipboardList },
  { path: "/analytics", key: "analytics", icon: Activity },
  { path: "/plan", key: "plan", icon: Map },
  { path: "/reports", key: "reports", icon: FileText },
  { path: "/settings", key: "settings", icon: Settings },
] as const;
const Auth = createContext<{
  session: Session | null;
  setSession: (s: Session | null) => void;
}>({ session: null, setSession: () => {} });

function Login() {
  const { session, setSession } = useContext(Auth);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(false);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  if (session) return <Navigate to="/" replace />;
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setPending(true);
    setError("");
    try {
      const result = await api<Session>("/api/v1/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password }),
      });
      saveSession(result, remember);
      setSession(result);
    } catch (e) {
      setError(e instanceof TypeError ? ru.loginError : (e as Error).message);
    } finally {
      setPending(false);
    }
  }
  return (
    <div className="login-page">
      <div className="login-story">
        <Brand />
        <div>
          <span className="eyebrow">{ru.brandSub}</span>
          <h1>{ru.stageTitle}</h1>
          <p>{ru.aboutText}</p>
          <div className="architecture">
            <Camera size={26} />
            <span>→</span>
            <Activity size={26} />
            <span>→</span>
            <ShieldCheck size={26} />
          </div>
        </div>
        <Link to="/about">
          {ru.about} <ArrowRight size={16} />
        </Link>
      </div>
      <main className="login-form">
        <span className="badge">{ru.stage}</span>
        <h2>{ru.login}</h2>
        <p className="muted">{ru.loginSub}</p>
        <form onSubmit={submit}>
          <label>
            {ru.username}
            <input
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              maxLength={100}
            />
          </label>
          <label>
            {ru.password}
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              maxLength={256}
            />
          </label>
          <label className="check-label">
            <input
              type="checkbox"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
            />
            {ru.remember}
          </label>
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
          <Button className="w-full" disabled={pending}>
            {pending ? ru.signingIn : ru.signIn}
            <ArrowRight size={17} />
          </Button>
        </form>
        <div className="demo-logins">
          <p>{ru.demoAccounts}</p>
          {Object.entries(ru.roles).map(([role, label]) => (
            <button
              key={role}
              onClick={() => {
                setUsername(role);
                setPassword(role);
              }}
            >
              <span>{label}</span>
              <code>
                {role} / {role}
              </code>
            </button>
          ))}
        </div>
        <small className="muted">{ru.restricted}</small>
      </main>
    </div>
  );
}
function Brand() {
  return (
    <div className="brand">
      <span className="brand-icon">
        <ShieldCheck size={25} />
      </span>
      <div>
        {ru.brand}
        <small>{ru.brandSub}</small>
      </div>
    </div>
  );
}
function Protected() {
  const { session } = useContext(Auth);
  return session ? (
    <RealtimeSession key={session.user.id} session={session}>
      <Outlet />
    </RealtimeSession>
  ) : (
    <Navigate to="/login" replace />
  );
}
function Layout() {
  const { session, setSession } = useContext(Auth);
  const enterprise = useQuery({
    queryKey: ["enterprise-settings", session?.user.id],
    queryFn: () =>
      api<{ enterprise_name: string; workshop_name: string }>(
        "/api/v1/settings",
      ),
  });
  const [mobile, setMobile] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const [globalSearch, setGlobalSearch] = useState("");
  const current = navigation.find((n) =>
    n.path === "/"
      ? location.pathname === "/"
      : location.pathname.startsWith(n.path),
  );
  return (
    <div className="shell">
      <aside className={mobile ? "sidebar mobile-open" : "sidebar"}>
        <Brand />
        <button
          className="mobile-close icon-button"
          aria-label={ru.back}
          onClick={() => setMobile(false)}
        >
          <X />
        </button>
        <div className="plant">
          <span className="status-dot" />
          {enterprise.data?.workshop_name ?? ru.enterprise}
        </div>
        <nav>
          {navigation.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === "/"}
              onClick={() => setMobile(false)}
            >
              <item.icon size={19} />
              {ru[item.key]}
              <ChevronRight className="nav-arrow" size={15} />
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <Link to="/wall">
            <Monitor size={18} />
            {ru.wall}
            <ArrowDownLeft size={15} />
          </Link>
          <Link to="/about">
            <CircleHelp size={18} />
            {ru.about}
          </Link>
          <div className="profile">
            <div className="avatar">{session?.user.full_name.slice(0, 1)}</div>
            <div>
              <strong>{session?.user.full_name}</strong>
              <small>{session && ru.roles[session.user.role]}</small>
            </div>
            <button
              aria-label={ru.logout}
              title={ru.logout}
              className="icon-button"
              onClick={() => {
                clearSession();
                setSession(null);
              }}
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header>
          <button
            className="mobile-menu icon-button"
            aria-label={ru.dashboard}
            onClick={() => setMobile(true)}
          >
            <Menu />
          </button>
          <span className="breadcrumb">
            {enterprise.data?.enterprise_name ?? ru.enterprise}
            <ChevronRight size={14} />
            <strong>{current ? ru[current.key] : ru.notFound}</strong>
          </span>
          <div className="header-actions">
            <form
              className="global-search"
              onSubmit={(event) => {
                event.preventDefault();
                navigate(
                  `/incidents?q=${encodeURIComponent(globalSearch.trim())}`,
                );
              }}
            >
              <Search size={17} />
              <input
                aria-label={ru.incidentUi.search}
                placeholder={ru.incidentUi.search}
                maxLength={250}
                value={globalSearch}
                onChange={(e) => setGlobalSearch(e.target.value)}
              />
            </form>
            <ConnectionStatus />
            <DemoBadge />
            <SoundButton />
            <NotificationCenter />
          </div>
        </header>
        <main className="content">
          <Outlet />
        </main>
        <footer>
          <span>
            {ru.brand} · {ru.brandSub.toLowerCase()}
          </span>
          <span>{ru.stage}</span>
        </footer>
      </div>
    </div>
  );
}
function Placeholder({ title }: { title: string }) {
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">{ru.placeholder}</span>
          <h1>{title}</h1>
        </div>
        <span className="badge">{ru.stage}</span>
      </div>
      <div className="empty panel">
        <div className="empty-icon">
          <Monitor size={34} />
        </div>
        <h2>{ru.emptyTitle}</h2>
        <p>{ru.emptyDescription}</p>
        <Button asChild variant="outline">
          <Link to="/">
            {ru.back}
            <ArrowRight size={16} />
          </Link>
        </Button>
      </div>
    </>
  );
}
function App() {
  const location = useLocation();
  const [session, setSession] = useState(readSession);
  const [checking, setChecking] = useState(!!session);
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const reset = () => setSession(null);
    window.addEventListener("session-expired", reset);
    return () => window.removeEventListener("session-expired", reset);
  }, []);
  useEffect(() => {
    if (!readSession()) {
      setChecking(false);
      return;
    }
    let active = true;
    setChecking(true);
    setFailed(false);
    api<User>("/api/v1/auth/me")
      .then((user) => {
        if (active) setSession((s) => (s ? { ...s, user } : s));
      })
      .catch(() => {
        if (active && readSession()) setFailed(true);
        else if (active) setSession(null);
      })
      .finally(() => {
        if (active) setChecking(false);
      });
    return () => {
      active = false;
    };
  }, [attempt]);
  if (
    (checking || failed) &&
    location.pathname.replace(/\/+$/, "") !== "/about"
  )
    return (
      <main className="public-page">
        <Brand />
        <p>{failed ? ru.error : ru.loading}</p>
        {failed && (
          <Button onClick={() => setAttempt((x) => x + 1)}>{ru.retry}</Button>
        )}
      </main>
    );
  return (
    <Auth.Provider value={{ session, setSession }}>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/about"
          element={
            <Suspense fallback={<div className="skeleton">{ru.loading}</div>}>
              <AboutPage />
            </Suspense>
          }
        />
        <Route element={<Protected />}>
          <Route
            path="/wall"
            element={
              <Suspense fallback={<div className="skeleton">{ru.loading}</div>}>
                <WallPage />
              </Suspense>
            }
          />
          <Route element={<Layout />}>
            <Route
              index
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <DashboardPage />
                </Suspense>
              }
            />
            <Route
              path="/analytics"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <AnalyticsPage />
                </Suspense>
              }
            />
            <Route
              path="/plan"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <PlanPage />
                </Suspense>
              }
            />
            <Route
              path="/reports"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <ReportsPage />
                </Suspense>
              }
            />
            <Route
              path="/live"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <LivePage />
                </Suspense>
              }
            />
            <Route path="/cameras" element={<CamerasPage />} />
            <Route path="/incidents" element={<IncidentsPage />} />
            <Route
              path="/settings"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <SettingsPage />
                </Suspense>
              }
            />
            <Route
              path="/reports/:reportId"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <ReportsPage />
                </Suspense>
              }
            />
            <Route
              path="/incidents/:id/act"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <ActPage />
                </Suspense>
              }
            />
            <Route
              path="/live/:cameraId"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <LivePage />
                </Suspense>
              }
            />
            <Route path="/cameras/:id" element={<CameraPage />} />
            <Route
              path="/cameras/:id/zones"
              element={
                <Suspense
                  fallback={<div className="skeleton">{ru.loading}</div>}
                >
                  <ZoneEditor />
                </Suspense>
              }
            />
            <Route path="/incidents/:id" element={<IncidentPage />} />
            <Route path="*" element={<Placeholder title={ru.notFound} />} />
          </Route>
        </Route>
      </Routes>
    </Auth.Provider>
  );
}
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
