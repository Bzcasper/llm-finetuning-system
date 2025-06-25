// API utilities for real-time data fetching
import { useState, useEffect, useCallback, useRef } from 'react'

export const API_BASE_URL = 'http://localhost:8000/api'

// Custom hook for polling API data
export function usePolling(fetchFunction, interval = 2000, dependencies = []) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)
  const mountedRef = useRef(true)

  const fetchData = useCallback(async () => {
    if (!mountedRef.current) return
    
    try {
      const result = await fetchFunction()
      if (mountedRef.current) {
        setData(result)
        setError(null)
      }
    } catch (err) {
      if (mountedRef.current) {
        setError(err)
        console.error('Polling error:', err)
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false)
      }
    }
  }, [fetchFunction])

  useEffect(() => {
    mountedRef.current = true
    
    // Initial fetch
    fetchData()
    
    // Set up polling
    intervalRef.current = setInterval(fetchData, interval)
    
    return () => {
      mountedRef.current = false
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [fetchData, interval, ...dependencies])

  const refetch = useCallback(() => {
    fetchData()
  }, [fetchData])

  return { data, loading, error, refetch }
}

// API functions
export const apiClient = {
  // Health check
  checkHealth: async () => {
    const response = await fetch(`${API_BASE_URL}/health`)
    if (!response.ok) throw new Error('Health check failed')
    return response.json()
  },

  // Training status
  getTrainingStatus: async (jobId) => {
    if (!jobId) return null
    const response = await fetch(`${API_BASE_URL}/training/status/${jobId}`)
    if (!response.ok) throw new Error('Failed to fetch training status')
    return response.json()
  },

  // Training logs
  getTrainingLogs: async (jobId) => {
    if (!jobId) return { logs: [] }
    const response = await fetch(`${API_BASE_URL}/training/logs/${jobId}`)
    if (!response.ok) throw new Error('Failed to fetch training logs')
    return response.json()
  },

  // System metrics
  getSystemMetrics: async () => {
    const response = await fetch(`${API_BASE_URL}/metrics/system`)
    if (!response.ok) throw new Error('Failed to fetch system metrics')
    return response.json()
  },

  // GPU metrics
  getGPUMetrics: async () => {
    const response = await fetch(`${API_BASE_URL}/metrics/gpu`)
    if (!response.ok) throw new Error('Failed to fetch GPU metrics')
    return response.json()
  },

  // Cost metrics
  getCostMetrics: async () => {
    const response = await fetch(`${API_BASE_URL}/metrics/cost`)
    if (!response.ok) throw new Error('Failed to fetch cost metrics')
    return response.json()
  },

  // All training jobs
  getAllJobs: async () => {
    const response = await fetch(`${API_BASE_URL}/training/jobs`)
    if (!response.ok) throw new Error('Failed to fetch training jobs')
    return response.json()
  },

  // Start training
  startTraining: async (config) => {
    const response = await fetch(`${API_BASE_URL}/training/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    })
    if (!response.ok) throw new Error('Failed to start training')
    return response.json()
  },

  // Stop training
  stopTraining: async (jobId) => {
    const response = await fetch(`${API_BASE_URL}/training/stop/${jobId}`, {
      method: 'POST',
    })
    if (!response.ok) throw new Error('Failed to stop training')
    return response.json()
  },

  // Export metrics
  exportMetrics: async (jobId, format = 'json') => {
    const response = await fetch(`${API_BASE_URL}/training/export/${jobId}?format=${format}`)
    if (!response.ok) throw new Error('Failed to export metrics')
    return response.json()
  }
}

// Data cache for performance optimization
class DataCache {
  constructor(ttl = 5000) { // 5 second TTL
    this.cache = new Map()
    this.ttl = ttl
  }

  get(key) {
    const item = this.cache.get(key)
    if (!item) return null
    
    if (Date.now() - item.timestamp > this.ttl) {
      this.cache.delete(key)
      return null
    }
    
    return item.data
  }

  set(key, data) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    })
  }

  clear() {
    this.cache.clear()
  }
}

export const dataCache = new DataCache()

// Cached API client with automatic caching
export const cachedApiClient = {
  getWithCache: async (key, fetchFunction) => {
    const cached = dataCache.get(key)
    if (cached) return cached
    
    const data = await fetchFunction()
    dataCache.set(key, data)
    return data
  }
}

// Error retry logic
export const withRetry = async (fn, maxRetries = 3, delay = 1000) => {
  let lastError
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error
      if (i < maxRetries - 1) {
        await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)))
      }
    }
  }
  
  throw lastError
}

// Format utilities
export const formatUtils = {
  formatBytes: (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  },

  formatTime: (seconds) => {
    if (!seconds) return 'N/A'
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`
    } else {
      return `${secs}s`
    }
  },

  formatCurrency: (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 4
    }).format(amount)
  },

  formatPercentage: (value) => {
    return `${(value || 0).toFixed(1)}%`
  },

  formatNumber: (value, decimals = 2) => {
    return (value || 0).toFixed(decimals)
  }
}

// WebSocket connection for real-time updates (optional enhancement)
export class WebSocketClient {
  constructor(url) {
    this.url = url
    this.ws = null
    this.listeners = new Map()
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
    this.reconnectDelay = 1000
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url)
      
      this.ws.onopen = () => {
        console.log('WebSocket connected')
        this.reconnectAttempts = 0
      }
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          const { type, payload } = data
          
          if (this.listeners.has(type)) {
            this.listeners.get(type).forEach(callback => callback(payload))
          }
        } catch (error) {
          console.error('WebSocket message parsing error:', error)
        }
      }
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected')
        this.attemptReconnect()
      }
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    } catch (error) {
      console.error('WebSocket connection error:', error)
      this.attemptReconnect()
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      setTimeout(() => {
        console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
        this.connect()
      }, this.reconnectDelay * this.reconnectAttempts)
    }
  }

  on(eventType, callback) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, [])
    }
    this.listeners.get(eventType).push(callback)
  }

  off(eventType, callback) {
    if (this.listeners.has(eventType)) {
      const callbacks = this.listeners.get(eventType)
      const index = callbacks.indexOf(callback)
      if (index > -1) {
        callbacks.splice(index, 1)
      }
    }
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
    }
  }
}