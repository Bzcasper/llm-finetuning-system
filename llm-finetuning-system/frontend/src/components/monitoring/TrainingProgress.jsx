import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Alert, AlertDescription } from '@/components/ui/alert.jsx'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  ComposedChart,
  Bar,
  Legend,
  ScatterChart,
  Scatter
} from 'recharts'
import { 
  Brain, 
  TrendingDown, 
  TrendingUp, 
  Clock, 
  Target,
  Activity,
  BarChart3,
  Download,
  Play,
  Pause,
  Square,
  CheckCircle,
  AlertTriangle,
  Info
} from 'lucide-react'
import { usePolling, apiClient, formatUtils } from '@/utils/api.js'

export default function TrainingProgress({ jobId, onJobAction }) {
  const [trainingHistory, setTrainingHistory] = useState([])
  const [milestones, setMilestones] = useState([])
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Fetch training status with polling
  const { data: trainingStatus, loading, error } = usePolling(
    () => jobId ? apiClient.getTrainingStatus(jobId) : null,
    2000,
    [jobId]
  )

  // Update training history
  useEffect(() => {
    if (trainingStatus) {
      const newDataPoint = {
        epoch: trainingStatus.current_epoch || 0,
        step: trainingStatus.current_step || 0,
        loss: trainingStatus.loss || 0,
        accuracy: trainingStatus.accuracy || 0,
        learning_rate: trainingStatus.learning_rate || 0,
        perplexity: trainingStatus.perplexity || 0,
        gradient_norm: trainingStatus.gradient_norm || 0,
        timestamp: new Date().toISOString(),
        time: Date.now()
      }

      setTrainingHistory(prev => {
        const filtered = prev.filter(item => 
          item.epoch !== newDataPoint.epoch || item.step !== newDataPoint.step
        )
        return [...filtered, newDataPoint].sort((a, b) => a.epoch - b.epoch || a.step - b.step)
      })

      // Check for milestones
      if (trainingStatus.current_epoch > 0 && trainingStatus.current_epoch % 5 === 0) {
        const milestone = {
          epoch: trainingStatus.current_epoch,
          loss: trainingStatus.loss,
          accuracy: trainingStatus.accuracy,
          timestamp: Date.now(),
          type: 'epoch_milestone'
        }
        
        setMilestones(prev => {
          const exists = prev.some(m => m.epoch === milestone.epoch && m.type === milestone.type)
          if (!exists) {
            return [...prev, milestone].sort((a, b) => b.epoch - a.epoch)
          }
          return prev
        })
      }
    }
  }, [trainingStatus])

  const getLossChangeRate = () => {
    if (trainingHistory.length < 2) return 0
    const recent = trainingHistory.slice(-2)
    return ((recent[1].loss - recent[0].loss) / recent[0].loss) * 100
  }

  const getEstimatedCompletion = () => {
    if (!trainingStatus || !trainingStatus.total_epochs) return null
    
    const progress = trainingStatus.current_epoch / trainingStatus.total_epochs
    const elapsed = Date.now() - (trainingStatus.start_time || Date.now())
    const totalEstimated = elapsed / progress
    const remaining = totalEstimated - elapsed
    
    return remaining > 0 ? remaining : 0
  }

  const getBestMetrics = () => {
    if (trainingHistory.length === 0) return { bestLoss: 0, bestAccuracy: 0, bestEpoch: 0 }
    
    const bestLoss = Math.min(...trainingHistory.map(h => h.loss))
    const bestAccuracy = Math.max(...trainingHistory.map(h => h.accuracy))
    const bestLossEntry = trainingHistory.find(h => h.loss === bestLoss)
    
    return {
      bestLoss,
      bestAccuracy,
      bestEpoch: bestLossEntry?.epoch || 0
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'bg-blue-500'
      case 'completed': return 'bg-green-500'
      case 'failed': return 'bg-red-500'
      case 'paused': return 'bg-yellow-500'
      default: return 'bg-gray-500'
    }
  }

  const handleJobAction = (action) => {
    if (onJobAction) {
      onJobAction(jobId, action)
    }
  }

  const exportData = async () => {
    try {
      const data = await apiClient.exportMetrics(jobId, 'json')
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `training_metrics_${jobId}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export failed:', error)
    }
  }

  if (!jobId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Training Progress
          </CardTitle>
          <CardDescription>No active training job</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8">
            <p className="text-gray-500">Start a training job to monitor progress</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (loading && !trainingStatus) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Training Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-32">
            <div className="text-sm text-gray-500">Loading training progress...</div>
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
            <Brain className="w-5 h-5" />
            Training Progress
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Failed to load training progress: {error.message}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  const bestMetrics = getBestMetrics()
  const lossChangeRate = getLossChangeRate()
  const estimatedCompletion = getEstimatedCompletion()

  return (
    <div className="space-y-6">
      {/* Training Overview */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Brain className="w-5 h-5" />
              Training Progress
              <div className={`w-2 h-2 rounded-full ${getStatusColor(trainingStatus?.status)}`} />
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                {showAdvanced ? 'Simple' : 'Advanced'}
              </Button>
              <Button size="sm" variant="outline" onClick={exportData}>
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </CardTitle>
          <CardDescription>
            Job ID: {jobId} • Status: {trainingStatus?.status || 'Unknown'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {trainingStatus?.current_epoch || 0}
              </div>
              <div className="text-sm text-gray-600">Current Epoch</div>
              <div className="text-xs text-gray-500">
                of {trainingStatus?.total_epochs || 0}
              </div>
            </div>
            
            <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <div className="text-2xl font-bold text-red-600">
                {formatUtils.formatNumber(trainingStatus?.loss)}
              </div>
              <div className="text-sm text-gray-600">Current Loss</div>
              <div className="flex items-center justify-center gap-1 text-xs">
                {lossChangeRate < 0 ? (
                  <TrendingDown className="w-3 h-3 text-green-500" />
                ) : (
                  <TrendingUp className="w-3 h-3 text-red-500" />
                )}
                <span className={lossChangeRate < 0 ? 'text-green-500' : 'text-red-500'}>
                  {formatUtils.formatNumber(Math.abs(lossChangeRate))}%
                </span>
              </div>
            </div>
            
            <div className="text-center p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {formatUtils.formatPercentage(trainingStatus?.accuracy * 100)}
              </div>
              <div className="text-sm text-gray-600">Accuracy</div>
              <div className="text-xs text-gray-500">
                Best: {formatUtils.formatPercentage(bestMetrics.bestAccuracy * 100)}
              </div>
            </div>
            
            <div className="text-center p-4 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {estimatedCompletion ? formatUtils.formatTime(estimatedCompletion / 1000) : 'N/A'}
              </div>
              <div className="text-sm text-gray-600">Est. Remaining</div>
              <div className="text-xs text-gray-500">
                {formatUtils.formatPercentage(trainingStatus?.progress)}
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Training Progress</span>
              <span>{formatUtils.formatPercentage(trainingStatus?.progress)}</span>
            </div>
            <Progress value={trainingStatus?.progress || 0} className="w-full" />
          </div>

          {/* Job Controls */}
          <div className="flex gap-2 mt-4">
            <Button 
              size="sm" 
              disabled={trainingStatus?.status !== 'running'}
              onClick={() => handleJobAction('pause')}
            >
              <Pause className="w-4 h-4 mr-2" />
              Pause
            </Button>
            <Button 
              size="sm" 
              disabled={trainingStatus?.status !== 'paused'}
              onClick={() => handleJobAction('resume')}
            >
              <Play className="w-4 h-4 mr-2" />
              Resume
            </Button>
            <Button 
              size="sm" 
              variant="destructive"
              disabled={!['running', 'paused'].includes(trainingStatus?.status)}
              onClick={() => handleJobAction('stop')}
            >
              <Square className="w-4 h-4 mr-2" />
              Stop
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Loss and Accuracy Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Training Loss</CardTitle>
            <CardDescription>
              Loss progression over epochs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip 
                  formatter={(value) => [formatUtils.formatNumber(value, 4), 'Loss']}
                  labelFormatter={(label) => `Epoch: ${label}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  dot={{ fill: '#ef4444', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Training Accuracy</CardTitle>
            <CardDescription>
              Accuracy improvement over epochs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trainingHistory}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis domain={[0, 1]} />
                <Tooltip 
                  formatter={(value) => [formatUtils.formatPercentage(value * 100), 'Accuracy']}
                  labelFormatter={(label) => `Epoch: ${label}`}
                />
                <Line 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="#22c55e" 
                  strokeWidth={2}
                  dot={{ fill: '#22c55e', r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {showAdvanced && (
        <>
          {/* Advanced Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Advanced Training Metrics</CardTitle>
              <CardDescription>
                Comprehensive training analytics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <ComposedChart data={trainingHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" />
                  <YAxis yAxisId="loss" orientation="left" />
                  <YAxis yAxisId="lr" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Area 
                    yAxisId="loss"
                    type="monotone" 
                    dataKey="loss" 
                    fill="#ef4444" 
                    fillOpacity={0.2}
                    stroke="#ef4444"
                    name="Loss"
                  />
                  <Line 
                    yAxisId="lr"
                    type="monotone" 
                    dataKey="learning_rate" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    name="Learning Rate"
                  />
                  <Bar 
                    yAxisId="loss"
                    dataKey="gradient_norm" 
                    fill="#8884d8" 
                    fillOpacity={0.6}
                    name="Gradient Norm"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Perplexity and Learning Rate */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Perplexity</CardTitle>
                <CardDescription>Model perplexity over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip formatter={(value) => [formatUtils.formatNumber(value), 'Perplexity']} />
                    <Area 
                      type="monotone" 
                      dataKey="perplexity" 
                      stroke="#f59e0b" 
                      fill="#f59e0b"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Learning Rate Schedule</CardTitle>
                <CardDescription>Learning rate changes over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip formatter={(value) => [value.toExponential(3), 'Learning Rate']} />
                    <Line 
                      type="monotone" 
                      dataKey="learning_rate" 
                      stroke="#8b5cf6" 
                      strokeWidth={2}
                      dot={{ fill: '#8b5cf6' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </>
      )}

      {/* Milestones */}
      {milestones.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5" />
              Training Milestones
            </CardTitle>
            <CardDescription>
              Key achievements during training
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {milestones.slice(0, 5).map((milestone, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <div>
                      <p className="font-medium">Epoch {milestone.epoch} Completed</p>
                      <p className="text-sm text-gray-500">
                        {new Date(milestone.timestamp).toLocaleString()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">
                      Loss: {formatUtils.formatNumber(milestone.loss, 4)}
                    </p>
                    <p className="text-sm text-gray-500">
                      Acc: {formatUtils.formatPercentage(milestone.accuracy * 100)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Training Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Training Summary</CardTitle>
          <CardDescription>
            Key statistics and performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-3">
              <h4 className="font-semibold text-sm text-gray-600">PERFORMANCE</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Best Loss:</span>
                  <span className="text-sm font-medium">
                    {formatUtils.formatNumber(bestMetrics.bestLoss, 4)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Best Accuracy:</span>
                  <span className="text-sm font-medium">
                    {formatUtils.formatPercentage(bestMetrics.bestAccuracy * 100)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Current LR:</span>
                  <span className="text-sm font-medium">
                    {trainingStatus?.learning_rate?.toExponential(3) || 'N/A'}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold text-sm text-gray-600">PROGRESS</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Epochs:</span>
                  <span className="text-sm font-medium">
                    {trainingStatus?.current_epoch || 0} / {trainingStatus?.total_epochs || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Steps:</span>
                  <span className="text-sm font-medium">
                    {trainingStatus?.current_step || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Completion:</span>
                  <span className="text-sm font-medium">
                    {formatUtils.formatPercentage(trainingStatus?.progress)}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="space-y-3">
              <h4 className="font-semibold text-sm text-gray-600">TIMING</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Elapsed:</span>
                  <span className="text-sm font-medium">
                    {trainingStatus?.elapsed_time ? formatUtils.formatTime(trainingStatus.elapsed_time) : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">Remaining:</span>
                  <span className="text-sm font-medium">
                    {estimatedCompletion ? formatUtils.formatTime(estimatedCompletion / 1000) : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm">ETA:</span>
                  <span className="text-sm font-medium">
                    {estimatedCompletion ? new Date(Date.now() + estimatedCompletion).toLocaleTimeString() : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}