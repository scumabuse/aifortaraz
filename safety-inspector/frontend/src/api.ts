export type Role = "admin" | "engineer" | "master" | "viewer";
export interface User {
  id: string;
  username: string;
  full_name: string;
  role: Role;
  area_id: string | null;
}
export interface Session {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: User;
}
export interface Health {
  status: string;
  database: string;
  connected_ai_modules: number;
  stage: number;
}
export interface DictionaryItem {
  code?: string;
  id?: string;
  name?: string;
  name_ru?: string;
}
export function readSession(): Session | null {
  try {
    const raw =
      localStorage.getItem("session") ?? sessionStorage.getItem("session");
    return raw ? (JSON.parse(raw) as Session) : null;
  } catch {
    return null;
  }
}
export function saveSession(value: Session, remember: boolean) {
  localStorage.removeItem("session");
  sessionStorage.removeItem("session");
  (remember ? localStorage : sessionStorage).setItem(
    "session",
    JSON.stringify(value),
  );
}
export function clearSession() {
  localStorage.removeItem("session");
  sessionStorage.removeItem("session");
}
let refreshing: Promise<Session> | null = null;
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
async function authorizedResponse(
  path: string,
  init: RequestInit = {},
  retry = true,
): Promise<Response> {
  const session = readSession();
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...(session ? { Authorization: `Bearer ${session.access_token}` } : {}),
      ...init.headers,
    },
  });
  if (
    response.status === 401 &&
    session &&
    retry &&
    path !== "/api/v1/auth/login" &&
    path !== "/api/v1/auth/refresh"
  ) {
    refreshing ??= api<Session>(
      "/api/v1/auth/refresh",
      {
        method: "POST",
        body: JSON.stringify({ refresh_token: session.refresh_token }),
      },
      false,
    ).finally(() => {
      refreshing = null;
    });
    try {
      const value = await refreshing;
      saveSession(value, localStorage.getItem("session") !== null);
    } catch (error) {
      if (error instanceof ApiError && [401, 403].includes(error.status)) {
        clearSession();
        window.dispatchEvent(new Event("session-expired"));
      }
      throw error;
    }
    return authorizedResponse(path, init, false);
  }
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      error?: { message?: string };
    } | null;
    throw new ApiError(
      body?.error?.message ?? `HTTP ${response.status}`,
      response.status,
    );
  }
  return response;
}

export async function api<T>(
  path: string,
  init: RequestInit = {},
  retry = true,
): Promise<T> {
  return (await authorizedResponse(path, init, retry)).json() as Promise<T>;
}
export async function apiBlob(
  path: string,
  init: RequestInit = {},
): Promise<Blob> {
  return (await authorizedResponse(path, init)).blob();
}
