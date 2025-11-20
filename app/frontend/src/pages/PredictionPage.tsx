import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { 
  Container, 
  Box, 
  Typography, 
  Button, 
  Slider, 
  LinearProgress, 
  Card, 
  CardContent,
  Alert,
  IconButton,
  Chip
} from '@mui/material'
import { ArrowBack, PlayArrow, Stop, Speed } from '@mui/icons-material'
import { motion } from 'framer-motion'
import SmileyAnimation from '../components/SmileyAnimation'
import PredictionIndicator from '../components/PredictionIndicator'
import { PredictionSkeleton } from '../components/SkeletonLoader'
import { PredictionWebSocket, PredictionMessage } from '../services/websocket'

export default function PredictionPage() {
  const { subjectId, runId } = useParams<{ subjectId: string; runId: string }>()
  const [ws, setWs] = useState<PredictionWebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const [predicting, setPredicting] = useState(false)
  const [currentPrediction, setCurrentPrediction] = useState<number | null>(null)
  const [currentConfidence, setCurrentConfidence] = useState<number | null>(null)
  const [currentGroundTruth, setCurrentGroundTruth] = useState<number | null>(null)
  const [currentCorrect, setCurrentCorrect] = useState<boolean | null>(null)
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [finalResults, setFinalResults] = useState<{ accuracy: number; correct: number; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [simulationSpeed, setSimulationSpeed] = useState<number>(1.0)
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()
  const wsRef = useRef<PredictionWebSocket | null>(null)

  useEffect(() => {
    if (!subjectId || !runId) return

    const websocket = new PredictionWebSocket(
      subjectId,
      runId,
      handleMessage,
      handleError
    )

    wsRef.current = websocket
    setWs(websocket)

    websocket.connect()
      .then(() => {
        setConnected(true)
        setError(null)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })

    return () => {
      if (wsRef.current) {
        wsRef.current.disconnect()
      }
    }
  }, [subjectId, runId])

  const handleMessage = (message: PredictionMessage) => {
    switch (message.type) {
      case 'connection':
        setConnected(true)
        break
      case 'status':
        break
      case 'prediction_start':
        setPredicting(true)
        setProgress({ current: 0, total: message.n_windows || 0 })
        setFinalResults(null)
        break
      case 'prediction':
        if (message.prediction !== undefined) {
          setCurrentPrediction(message.prediction)
          setCurrentConfidence(message.confidence || null)
          setCurrentGroundTruth(message.ground_truth !== undefined ? message.ground_truth : null)
          setCurrentCorrect(message.correct !== undefined ? message.correct : null)
          if (message.progress) {
            setProgress(message.progress)
          }
        }
        break
      case 'prediction_complete':
        setPredicting(false)
        if (message.accuracy !== undefined && message.correct !== undefined && message.total !== undefined) {
          setFinalResults({
            accuracy: message.accuracy,
            correct: message.correct,
            total: message.total
          })
        }
        break
      case 'error':
        setError(message.message || 'An error occurred')
        break
    }
  }

  const handleError = (err: Error) => {
    setError(err.message)
    setConnected(false)
  }

  const handleStart = () => {
    if (ws && connected) {
      ws.send({ type: 'start', speed: simulationSpeed })
    }
  }

  const handleStop = () => {
    if (ws && connected) {
      ws.send({ type: 'stop' })
    }
  }

  const handleViewResults = () => {
    if (subjectId && runId && finalResults) {
      navigate(`/subject/${subjectId}/run/${runId}/results`, {
        state: finalResults
      })
    }
  }

  if (loading) {
    return (
      <Container maxWidth="lg" className="min-h-screen py-12">
        <PredictionSkeleton />
      </Container>
    )
  }

  if (error && !connected) {
    return (
      <Container maxWidth="lg" className="min-h-screen py-12">
        <Alert 
          severity="error" 
          action={
            <Button color="inherit" size="small" onClick={() => navigate('/')}>
              Back to Subjects
            </Button>
          }
          className="bg-white/95 mb-4"
        >
          {error}
        </Alert>
      </Container>
    )
  }

  const progressPercentage = progress.total > 0 ? (progress.current / progress.total) * 100 : 0

  return (
    <Container maxWidth="lg" className="min-h-screen py-12">
      <Box className="mb-8">
        <motion.div
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 400, damping: 10 }}
        >
          <IconButton
            onClick={() => navigate(`/subject/${subjectId}/runs`)}
            className="mb-4"
            sx={{
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              color: '#3b82f6',
              '&:hover': {
                backgroundColor: 'rgba(59, 130, 246, 0.2)',
              },
            }}
          >
            <ArrowBack />
          </IconButton>
        </motion.div>
        <Typography 
          variant="h3" 
          className="font-bold mb-2"
          sx={{ 
            fontSize: { xs: '1.75rem', md: '2.5rem' },
            background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          Real-time Predictions
        </Typography>
        <Box className="flex items-center gap-2 mb-4">
          <Chip 
            label={`Subject ${subjectId}`} 
            sx={{
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              color: '#3b82f6',
              border: '1px solid rgba(59, 130, 246, 0.2)',
            }}
            size="small"
          />
          <Chip 
            label={`Run ${runId}`} 
            sx={{
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              color: '#3b82f6',
              border: '1px solid rgba(59, 130, 246, 0.2)',
            }}
            size="small"
          />
          {connected && (
            <Chip 
              label="Connected" 
              sx={{
                backgroundColor: 'rgba(34, 197, 94, 0.15)',
                color: '#16a34a',
              }}
              size="small"
            />
          )}
        </Box>
      </Box>

      {error && (
        <Alert severity="warning" className="bg-white/95 mb-6">
          {error}
        </Alert>
      )}

      <Box className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <Box className="lg:col-span-2">
          <Card 
            sx={{
              background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)',
              backdropFilter: 'blur(10px)',
              boxShadow: '0 10px 40px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04)',
              borderRadius: '20px',
              border: '1px solid rgba(255, 255, 255, 0.8)',
            }}
          >
            <CardContent className="p-8">
              <Box className="flex flex-col items-center justify-center min-h-[500px]">
                <SmileyAnimation 
                  prediction={currentPrediction} 
                  correct={currentCorrect}
                />
                <Box className="mt-10 w-full max-w-md">
                  <PredictionIndicator
                    prediction={currentPrediction}
                    confidence={currentConfidence}
                    groundTruth={currentGroundTruth}
                    correct={currentCorrect}
                  />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Box>

        <Box>
          <Card 
            sx={{
              background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)',
              backdropFilter: 'blur(10px)',
              boxShadow: '0 10px 40px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04)',
              borderRadius: '20px',
              border: '1px solid rgba(255, 255, 255, 0.8)',
            }}
            className="mb-6"
          >
            <CardContent className="p-6">
              <Box className="flex items-center gap-3 mb-6">
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)',
                  }}
                >
                  <Speed sx={{ color: 'white', fontSize: 24 }} />
                </Box>
                <Typography 
                  variant="h6" 
                  className="font-bold text-gray-800"
                  sx={{ fontSize: '1.25rem' }}
                >
                  Controls
                </Typography>
              </Box>
              
              {!predicting && !finalResults && (
                <Box className="space-y-6">
                  <Box
                    sx={{
                      p: 3,
                      borderRadius: '16px',
                      background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                      border: '1px solid rgba(59, 130, 246, 0.2)',
                    }}
                  >
                    <Box className="flex items-center gap-2 mb-3">
                      <Speed className="text-blue-600" sx={{ fontSize: 20 }} />
                      <Typography variant="body1" className="text-gray-800 font-semibold">
                        Simulation Speed
                      </Typography>
                    </Box>
                    <Slider
                      value={simulationSpeed}
                      onChange={(_, value) => setSimulationSpeed(value as number)}
                      min={0.5}
                      max={8}
                      step={0.5}
                      disabled={!connected}
                      sx={{
                        color: '#3b82f6',
                        '& .MuiSlider-thumb': {
                          width: 20,
                          height: 20,
                          boxShadow: '0 2px 8px rgba(59, 130, 246, 0.4)',
                        },
                        '& .MuiSlider-track': {
                          height: 6,
                          borderRadius: 3,
                        },
                        '& .MuiSlider-rail': {
                          height: 6,
                          borderRadius: 3,
                          opacity: 0.3,
                        },
                      }}
                      className="mb-2"
                    />
                    <Typography 
                      variant="body2" 
                      className="text-gray-600 font-medium"
                      sx={{ mt: 1 }}
                    >
                      {simulationSpeed === 1.0 && 'Real-time (4s per prediction)'}
                      {simulationSpeed > 1.0 && `${simulationSpeed}x faster (${(4.0 / simulationSpeed).toFixed(1)}s per prediction)`}
                      {simulationSpeed < 1.0 && `${simulationSpeed}x slower (${(4.0 / simulationSpeed).toFixed(1)}s per prediction)`}
                    </Typography>
                  </Box>
                  <motion.div
                    whileHover={{ scale: 1.02, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                    transition={{ type: "spring", stiffness: 400, damping: 10 }}
                  >
                    <Button
                      variant="contained"
                      fullWidth
                      size="large"
                      onClick={handleStart}
                      disabled={!connected}
                      startIcon={<PlayArrow />}
                      sx={{
                        py: 1.5,
                        background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                        borderRadius: '12px',
                        fontSize: '1rem',
                        fontWeight: 600,
                        textTransform: 'none',
                        boxShadow: '0 4px 16px rgba(59, 130, 246, 0.4)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #2563eb 0%, #1e40af 100%)',
                          boxShadow: '0 6px 20px rgba(59, 130, 246, 0.5)',
                          transform: 'translateY(-2px)',
                        },
                        '&:disabled': {
                          background: '#cbd5e1',
                          boxShadow: 'none',
                        },
                      }}
                    >
                      {connected ? 'Start Predictions' : 'Connecting...'}
                    </Button>
                  </motion.div>
                </Box>
              )}

              {predicting && (
                <Box className="space-y-6">
                  <motion.div
                    whileHover={{ scale: 1.02, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                    transition={{ type: "spring", stiffness: 400, damping: 10 }}
                  >
                    <Button
                      variant="contained"
                      fullWidth
                      size="large"
                      onClick={handleStop}
                      startIcon={<Stop />}
                      sx={{
                        py: 1.5,
                        background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                        borderRadius: '12px',
                        fontSize: '1rem',
                        fontWeight: 600,
                        textTransform: 'none',
                        boxShadow: '0 4px 16px rgba(239, 68, 68, 0.4)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)',
                          boxShadow: '0 6px 20px rgba(239, 68, 68, 0.5)',
                          transform: 'translateY(-2px)',
                        },
                      }}
                    >
                      Stop Predictions
                    </Button>
                  </motion.div>
                  <Box
                    sx={{
                      p: 3,
                      borderRadius: '16px',
                      background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                      border: '1px solid rgba(34, 197, 94, 0.2)',
                    }}
                  >
                    <Box className="flex justify-between items-center mb-3">
                      <Typography variant="body1" className="text-gray-800 font-semibold">
                        Progress
                      </Typography>
                      <Typography 
                        variant="body1" 
                        className="text-gray-700 font-bold"
                        sx={{ fontSize: '1.1rem' }}
                      >
                        {progress.current} / {progress.total}
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={progressPercentage}
                      sx={{
                        height: 10,
                        borderRadius: 5,
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          borderRadius: 5,
                          background: 'linear-gradient(90deg, #22c55e 0%, #16a34a 100%)',
                        },
                      }}
                    />
                    <Typography 
                      variant="caption" 
                      className="text-gray-600 mt-2 block text-center"
                      sx={{ fontSize: '0.875rem' }}
                    >
                      {progressPercentage.toFixed(1)}% Complete
                    </Typography>
                  </Box>
                </Box>
              )}

              {finalResults && (
                <Box className="space-y-6">
                  <Card 
                    sx={{
                      background: 'linear-gradient(135deg, #dcfce7 0%, #bbf7d0 50%, #86efac 100%)',
                      border: '2px solid #4ade80',
                      borderRadius: '16px',
                      boxShadow: '0 8px 24px rgba(34, 197, 94, 0.2)',
                    }}
                  >
                    <CardContent className="p-5">
                      <Typography 
                        variant="h6" 
                        sx={{ color: '#166534', fontWeight: 700 }} 
                        className="mb-3"
                      >
                        Predictions Complete! 🎉
                      </Typography>
                      <Typography 
                        variant="h3" 
                        sx={{ color: '#15803d', fontWeight: 800 }} 
                        className="mb-2"
                      >
                        {(finalResults.accuracy * 100).toFixed(1)}%
                      </Typography>
                      <Typography 
                        variant="body1" 
                        sx={{ color: '#15803d', fontWeight: 600 }}
                      >
                        {finalResults.correct} / {finalResults.total} correct predictions
                      </Typography>
                    </CardContent>
                  </Card>
                  <motion.div
                    whileHover={{ scale: 1.02, y: -2 }}
                    whileTap={{ scale: 0.98 }}
                    transition={{ type: "spring", stiffness: 400, damping: 10 }}
                  >
                    <Button
                      variant="contained"
                      fullWidth
                      size="large"
                      onClick={handleViewResults}
                      sx={{
                        py: 1.5,
                        background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
                        borderRadius: '12px',
                        fontSize: '1rem',
                        fontWeight: 600,
                        textTransform: 'none',
                        boxShadow: '0 4px 16px rgba(139, 92, 246, 0.4)',
                        '&:hover': {
                          background: 'linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%)',
                          boxShadow: '0 6px 20px rgba(139, 92, 246, 0.5)',
                          transform: 'translateY(-2px)',
                        },
                      }}
                    >
                      View Detailed Results
                    </Button>
                  </motion.div>
                </Box>
              )}
            </CardContent>
          </Card>
        </Box>
      </Box>
    </Container>
  )
}
