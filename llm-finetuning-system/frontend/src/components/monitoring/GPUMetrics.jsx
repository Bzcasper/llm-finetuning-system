import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  PieChart,
  Pie,
  Cell
} from 'recharts'
import { 
  Cpu, 
  Zap, 
  Thermometer, 
  Activity, 
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react'
import { usePolling, apiClient, formatUtils } from '@/utils/api.js'

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8']

export default function GPUMetrics({ jobId, showDetailed = true }) {
  const [historicalData, setHistoricalData] = useState([])
  const [alerts, setAlerts] = useState([])

  // Fetch GPU metrics with polling
  const { data: gpuMetrics, loading, error } = usePolling(
    () => apiClient.getGPUMetrics(),
    2000, // 2 second interval
    [jobId]
  )

  // Update historical data
  useEffect(() => {
    if (gpuMetrics) {
      const timestamp = new Date().toLocaleTimeString()
      const newDataPoint = {
        timestamp,
        time: Date.now(),
        utilization: gpuMetrics.utilization || 0,
        memory: gpuMetrics.memory_used || 0,
        temperature: gpuMetrics.temperature || 0,
        powerDraw: gpuMetrics.power_draw || 0,
        memoryTotal: gpuMetrics.memory_total || 0,
        clockSpeed: gpuMetrics.clock_speed || 0
      }

      setHistoricalData(prev => {
        const updated = [...prev, newDataPoint].slice(-50) // Keep last 50 points
        return updated
      })

      // Check for alerts
      const newAlerts = []
      if (gpuMetrics.utilization > 95) {
        newAlerts.push({
          type: 'warning',
          message: 'GPU utilization critically high (>95%)',
          timestamp: Date.now()
        })
      }
      if (gpuMetrics.temperature > 85) {
        newAlerts.push({
          type: 'error',
          message: 'GPU temperature critically high (>85°C)',
          timestamp: Date.now()
        })
      }
      if (gpuMetrics.memory_used / gpuMetrics.memory_total > 0.9) {
        newAlerts.push({
          type: 'warning',
          message: 'GPU memory usage high (>90%)',
          timestamp: Date.now()
        })
      }

      setAlerts(prev => [...newAlerts, ...prev].slice(0, 10)) // Keep last 10 alerts
    }
  }, [gpuMetrics])

  const getUtilizationTrend = () => {
    if (historicalData.length < 2) return 'stable'
    const recent = historicalData.slice(-5)
    const avg = recent.reduce((sum, d) => sum + d.utilization, 0) / recent.length
    const prevAvg = historicalData.slice(-10, -5).reduce((sum, d) => sum + d.utilization, 0) / 5
    
    if (avg > prevAvg + 5) return 'up'
    if (avg < prevAvg - 5) return 'down'
    return 'stable'
  }

  const getTrendIcon = (trend) => {
    switch (trend) {
      case 'up': return <TrendingUp className="w-4 h-4 text-green-500" />
      case 'down': return <TrendingDown className="w-4 h-4 text-red-500" />
      default: return <Minus className="w-4 h-4 text-gray-500" />
    }
  }

  const memoryData = gpuMetrics ? [
    { name: 'Used', value: gpuMetrics.memory_used || 0, color: '#FF8042' },
    { name: 'Free', value: (gpuMetrics.memory_total || 0) - (gpuMetrics.memory_used || 0), color: '#00C49F' }
  ] : []

  if (loading && !gpuMetrics) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            GPU Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <div className="text-sm text-gray-500">Loading GPU metrics...</div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            GPU Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Failed to load GPU metrics: {error.message}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-2">
          {alerts.slice(0, 3).map((alert, index) => (
            <Alert key={index} variant={alert.type === 'error' ? 'destructive' : 'default'}>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{alert.message}</AlertDescription>
            </Alert>
          ))}
        </div>
      )}

      {/* Current Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">GPU Utilization</p>
                <p className="text-2xl font-bold">
                  {formatUtils.formatPercentage(gpuMetrics?.utilization)}
                </p>
              </div>
              <div className="flex items-center gap-2">
                {getTrendIcon(getUtilizationTrend())}
                <Activity className="w-8 h-8 text-blue-500" />
              </div>
            </div>
            <Progress value={gpuMetrics?.utilization || 0} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Memory Usage</p>
                <p className="text-2xl font-bold">
                  {formatUtils.formatBytes((gpuMetrics?.memory_used || 0) * 1024 * 1024 * 1024)}
                </p>
                <p className="text-xs text-gray-500">
                  of {formatUtils.formatBytes((gpuMetrics?.memory_total || 0) * 1024 * 1024 * 1024)}
                </p>
              </div>
              <Cpu className="w-8 h-8 text-green-500" />
            </div>
            <Progress 
              value={gpuMetrics ? (gpuMetrics.memory_used / gpuMetrics.memory_total) * 100 : 0} 
              className="mt-2" 
            />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Temperature</p>
                <p className="text-2xl font-bold">
                  {gpuMetrics?.temperature || 0}°C
                </p>
              </div>
              <Thermometer className={`w-8 h-8 ${
                (gpuMetrics?.temperature || 0) > 80 ? 'text-red-500' : 
                (gpuMetrics?.temperature || 0) > 70 ? 'text-yellow-500' : 'text-blue-500'
              }`} />
            </div>
            <Progress 
              value={Math.min((gpuMetrics?.temperature || 0) / 100 * 100, 100)} 
              className="mt-2" 
            />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Power Draw</p>
                <p className="text-2xl font-bold">
                  {gpuMetrics?.power_draw || 0}W
                </p>
              </div>
              <Zap className="w-8 h-8 text-yellow-500" />
            </div>
            <Progress 
              value={Math.min((gpuMetrics?.power_draw || 0) / 400 * 100, 100)} 
              className="mt-2" 
            />
          </CardContent>
        </Card>
      </div>

      {showDetailed && (
        <>
          {/* Utilization Chart */}
          <Card>
            <CardHeader>
              <CardTitle>GPU Utilization Over Time</CardTitle>
              <CardDescription>Real-time GPU usage monitoring</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip 
                    formatter={(value) => [formatUtils.formatPercentage(value), 'Utilization']}
                    labelFormatter={(label) => `Time: ${label}`}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="utilization" 
                    stroke="#8884d8" 
                    fill="#8884d8"
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Memory Usage Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Memory Usage Over Time</CardTitle>
                <CardDescription>GPU memory consumption tracking</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value) => [formatUtils.formatBytes(value * 1024 * 1024 * 1024), 'Memory']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="memory" 
                      stroke="#00C49F" 
                      strokeWidth={2}
                      dot={{ fill: '#00C49F' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Memory Distribution</CardTitle>
                <CardDescription>Current memory allocation</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={memoryData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {memoryData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => formatUtils.formatBytes(value * 1024 * 1024 * 1024)} />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Temperature and Power */}
          <Card>
            <CardHeader>
              <CardTitle>Temperature & Power Monitoring</CardTitle>
              <CardDescription>Thermal and power consumption tracking</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historicalData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis yAxisId="temp" orientation="left" />
                  <YAxis yAxisId="power" orientation="right" />
                  <Tooltip />
                  <Line 
                    yAxisId="temp"
                    type="monotone" 
                    dataKey="temperature" 
                    stroke="#FF8042" 
                    strokeWidth={2}
                    name="Temperature (°C)"
                  />
                  <Line 
                    yAxisId="power"
                    type="monotone" 
                    dataKey="powerDraw" 
                    stroke="#FFBB28" 
                    strokeWidth={2}
                    name="Power (W)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* GPU Information */}
          <Card>
            <CardHeader>
              <CardTitle>GPU Information</CardTitle>
              <CardDescription>Hardware specifications and status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-600">Model</p>
                  <p className="text-lg font-semibold">{gpuMetrics?.model || 'Unknown'}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-600">Driver Version</p>
                  <p className="text-lg font-semibold">{gpuMetrics?.driver_version || 'Unknown'}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-600">CUDA Version</p>
                  <p className="text-lg font-semibold">{gpuMetrics?.cuda_version || 'Unknown'}</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-600">Clock Speed</p>
                  <p className="text-lg font-semibold">{gpuMetrics?.clock_speed || 0} MHz</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-600">Memory Clock</p>
                  <p className="text-lg font-semibold">{gpuMetrics?.memory_clock || 0} MHz</p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-600">Status</p>
                  <Badge variant={gpuMetrics?.status === 'active' ? 'default' : 'secondary'}>
                    {gpuMetrics?.status || 'Unknown'}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}