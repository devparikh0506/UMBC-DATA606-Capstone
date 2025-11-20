export interface PredictionMessage {
  type: 'connection' | 'status' | 'prediction' | 'prediction_start' | 'prediction_complete' | 'prediction_stopped' | 'error'
  status?: string
  message?: string
  subject_id?: string
  run_id?: string
  timestamp?: number
  prediction?: number
  confidence?: number
  ground_truth?: number
  correct?: boolean
  trial_idx?: number
  window_idx?: number
  progress?: {
    current: number
    total: number
  }
  accuracy?: number
  correct?: number
  total?: number
  n_trials?: number
  n_windows?: number
}

export class PredictionWebSocket {
  private ws: WebSocket | null = null
  private url: string
  private onMessage: (message: PredictionMessage) => void
  private onError: (error: Error) => void
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5

  constructor(
    subjectId: string,
    runId: string,
    onMessage: (message: PredictionMessage) => void,
    onError: (error: Error) => void
  ) {
    // Use ws:// for development, wss:// for production
    // In development, Vite proxy handles /ws, but WebSocket needs direct connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.hostname === 'localhost' ? 'localhost' : window.location.hostname
    const port = '8000' // Django default port
    this.url = `${protocol}//${host}:${port}/ws/predict/${subjectId}/${runId}/`
    this.onMessage = onMessage
    this.onError = onError
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)

        this.ws.onopen = () => {
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: PredictionMessage = JSON.parse(event.data)
            this.onMessage(message)
          } catch (error) {
            this.onError(new Error('Failed to parse WebSocket message'))
          }
        }

        this.ws.onerror = (error) => {
          this.onError(new Error('WebSocket error'))
        }

        this.ws.onclose = () => {
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++
            setTimeout(() => {
              this.connect().catch(reject)
            }, 1000 * this.reconnectAttempts)
          } else {
            this.onError(new Error('WebSocket connection closed'))
          }
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  send(message: { type: string; [key: string]: any }): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      this.onError(new Error('WebSocket is not connected'))
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }
}

