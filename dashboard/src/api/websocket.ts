import { Unit } from "../types/schemas";

export type WebSocketEventType =
  | "init"
  | "incident_update"
  | "decision_ready"
  | "decision_update"
  | "units_tick"
  | "scenario_state"
  | "hospital_prealert"
  | "green_corridor_active"
  | "pong";

export interface WebSocketEvent<T = any> {
  type: WebSocketEventType;
  data: T;
  timestamp?: string;
}

export interface UnitTickPayload {
  units: Unit[];
}

export type EventHandler = (event: WebSocketEvent) => void;

export class ResQWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Set<EventHandler> = new Set();
  private reconnectInterval = 3000;
  private pingInterval: any = null;
  private shouldReconnect = true;

  constructor(url?: string) {
    if (url) {
      this.url = url;
    } else {
      const loc = window.location;
      const proto = loc.protocol === "https:" ? "wss:" : "ws:";
      this.url = `${proto}//${loc.host}/ws`;
    }
  }

  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log("[ResQCity WS] Connected to live event bus");
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        try {
          const parsed: WebSocketEvent = JSON.parse(event.data);
          this.handlers.forEach((h) => h(parsed));
        } catch (e) {
          console.warn("[ResQCity WS] Message parse error", e);
        }
      };

      this.ws.onclose = () => {
        this.stopHeartbeat();
        if (this.shouldReconnect) {
          setTimeout(() => this.connect(), this.reconnectInterval);
        }
      };

      this.ws.onerror = (err) => {
        console.warn("[ResQCity WS] Connection warning:", err);
      };
    } catch (e) {
      console.warn("[ResQCity WS] Failed to initialize WebSocket:", e);
      if (this.shouldReconnect) {
        setTimeout(() => this.connect(), this.reconnectInterval);
      }
    }
  }

  public subscribe(handler: EventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  public send(type: string, data?: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }));
    }
  }

  public close(): void {
    this.shouldReconnect = false;
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.pingInterval = setInterval(() => {
      this.send("ping");
    }, 15000);
  }

  private stopHeartbeat(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }
}

export const wsClient = new ResQWebSocket();
