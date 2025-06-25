// API configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    }

    try {
      const response = await fetch(url, config)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error)
      throw error
    }
  }

  // Health check
  async checkHealth() {
    return this.request('/health')
  }

  // Dataset operations
  async getDatasets() {
    return this.request('/datasets')
  }

  async uploadDataset(file) {
    const formData = new FormData()
    formData.append('file', file)
    
    return this.request('/datasets/upload', {
      method: 'POST',
      body: formData,
      headers: {} // Remove Content-Type to let browser set it for FormData
    })
  }

  // Model operations
  async getModels() {
    return this.request('/models')
  }

  // Training operations
  async startTraining(config) {
    return this.request('/training/start', {
      method: 'POST',
      body: JSON.stringify(config)
    })
  }

  async getTrainingStatus(jobId) {
    return this.request(`/training/status/${jobId}`)
  }

  async getTrainingLogs(jobId) {
    return this.request(`/training/logs/${jobId}`)
  }

  async stopTraining(jobId) {
    return this.request(`/training/stop/${jobId}`, {
      method: 'POST'
    })
  }

  async pauseTraining(jobId) {
    return this.request(`/training/pause/${jobId}`, {
      method: 'POST'
    })
  }

  async resumeTraining(jobId) {
    return this.request(`/training/resume/${jobId}`, {
      method: 'POST'
    })
  }

  // Modal.com status
  async getModalStatus() {
    return this.request('/modal-status')
  }

  // Credentials management
  async saveCredentials(credentials) {
    return this.request('/credentials', {
      method: 'POST',
      body: JSON.stringify(credentials)
    })
  }

  async verifyCredentials() {
    return this.request('/credentials/verify')
  }
}

export const apiService = new ApiService()

// Export individual methods for easier import
export const {
  checkHealth,
  getDatasets,
  uploadDataset,
  getModels,
  startTraining,
  getTrainingStatus,
  getTrainingLogs,
  stopTraining,
  pauseTraining,
  resumeTraining,
  getModalStatus,
  saveCredentials,
  verifyCredentials
} = apiService