import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { 
  Brain, 
  Zap, 
  Database, 
  Server, 
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Play,
  Pause,
  Square
} from 'lucide-react'
import { formatTime } from '@/lib/utils.js'

export function Dashboard({ 
  trainingJobs = [], 
  currentJob, 
  onStartTraining, 
  onStopTraining, 
  onPauseTraining 
}) {
  const [systemStats, setSystemStats] = useState({
    totalJobs: 0,
    activeJobs: 0,
    completedJobs: 0,
    failedJobs: 0
  })

  useEffect(() => {
    const stats = trainingJobs.reduce((acc, job) => {
      acc.totalJobs++
      if (job.status === 'running') acc.activeJobs++
      else if (job.status === 'completed') acc.completedJobs++
      else if (job.status === 'failed') acc.failedJobs++
      return acc
    }, { totalJobs: 0, activeJobs: 0, completedJobs: 0, failedJobs: 0 })
    
    setSystemStats(stats)
  }, [trainingJobs])

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'running':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      case 'paused':
        return <Pause className="w-4 h-4 text-yellow-500" />
      default:
        return <Clock className="w-4 h-4 text-gray-500" />
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
      case 'running':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
      case 'paused':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-300'
    }
  }

  return (
    <div className="space-y-6">
      {/* System Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Jobs</p>
                <p className="text-2xl font-bold">{systemStats.totalJobs}</p>
              </div>
              <Brain className="w-8 h-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active</p>
                <p className="text-2xl font-bold text-blue-600">{systemStats.activeJobs}</p>
              </div>
              <Zap className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold text-green-600">{systemStats.completedJobs}</p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Failed</p>
                <p className="text-2xl font-bold text-red-600">{systemStats.failedJobs}</p>
              </div>
              <XCircle className="w-8 h-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Current Job Status */}
      {currentJob && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {getStatusIcon(currentJob.status)}
              Current Training Job
            </CardTitle>
            <CardDescription>
              Job ID: {currentJob.job_id} • Started: {new Date(currentJob.created_at).toLocaleString()}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm font-medium">
                  {currentJob.model_name} • {currentJob.gpu_type} GPU
                </p>
                <p className="text-xs text-muted-foreground">
                  Epoch {currentJob.current_epoch || 0} of {currentJob.total_epochs || 0}
                </p>
              </div>
              <Badge className={getStatusColor(currentJob.status)}>
                {currentJob.status}
              </Badge>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{currentJob.progress?.toFixed(1) || 0}%</span>
              </div>
              <Progress value={currentJob.progress || 0} className="w-full" />
            </div>

            {currentJob.estimated_time_remaining && (
              <p className="text-sm text-muted-foreground">
                Estimated time remaining: {formatTime(currentJob.estimated_time_remaining)}
              </p>
            )}

            <div className="flex gap-2">
              {currentJob.status === 'running' && (
                <>
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={() => onPauseTraining(currentJob.job_id)}
                  >
                    <Pause className="w-4 h-4 mr-2" />
                    Pause
                  </Button>
                  <Button 
                    size="sm" 
                    variant="destructive"
                    onClick={() => onStopTraining(currentJob.job_id)}
                  >
                    <Square className="w-4 h-4 mr-2" />
                    Stop
                  </Button>
                </>
              )}
              {currentJob.status === 'paused' && (
                <Button 
                  size="sm"
                  onClick={() => onStartTraining(currentJob.job_id)}
                >
                  <Play className="w-4 h-4 mr-2" />
                  Resume
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Jobs */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Training Jobs</CardTitle>
          <CardDescription>
            Overview of your recent fine-tuning jobs
          </CardDescription>
        </CardHeader>
        <CardContent>
          {trainingJobs.length === 0 ? (
            <div className="text-center py-8">
              <Database className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No training jobs yet</p>
              <p className="text-sm text-muted-foreground">
                Start your first fine-tuning job from the Training tab
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {trainingJobs.slice(0, 5).map((job) => (
                <div 
                  key={job.job_id}
                  className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <p className="font-medium">{job.model_name}</p>
                      <p className="text-sm text-muted-foreground">
                        {job.dataset_name} • {job.gpu_type}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <Badge className={getStatusColor(job.status)}>
                      {job.status}
                    </Badge>
                    <p className="text-xs text-muted-foreground mt-1">
                      {formatTime(job.duration)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>
            Common actions for managing your training jobs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button 
              variant="outline" 
              className="h-20 flex flex-col gap-2"
              onClick={() => onStartTraining()}
            >
              <Play className="w-5 h-5" />
              <span>Start New Job</span>
            </Button>
            <Button 
              variant="outline" 
              className="h-20 flex flex-col gap-2"
            >
              <Database className="w-5 h-5" />
              <span>Upload Dataset</span>
            </Button>
            <Button 
              variant="outline" 
              className="h-20 flex flex-col gap-2"
            >
              <Server className="w-5 h-5" />
              <span>View Logs</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}