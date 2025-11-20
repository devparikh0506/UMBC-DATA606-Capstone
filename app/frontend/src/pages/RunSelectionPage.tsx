import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Container, Typography, Box, Alert, Button, IconButton } from '@mui/material'
import { ArrowBack, Refresh } from '@mui/icons-material'
import { motion } from 'framer-motion'
import RunCard from '../components/RunCard'
import { RunCardSkeleton } from '../components/SkeletonLoader'
import { getRuns, Run } from '../services/api'

export default function RunSelectionPage() {
  const { subjectId } = useParams<{ subjectId: string }>()
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (subjectId) {
      loadRuns()
    }
  }, [subjectId])

  const loadRuns = async () => {
    if (!subjectId) return
    
    try {
      setLoading(true)
      const data = await getRuns(subjectId)
      setRuns(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load runs')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="xl" className="min-h-screen py-8 px-4">
      <Box className="mb-8">
        <motion.div
          whileHover={{ scale: 1.05 }}
          transition={{ type: "spring", stiffness: 400, damping: 10 }}
        >
          <IconButton
            onClick={() => navigate('/')}
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
            fontWeight: 700,
            letterSpacing: '-0.02em',
            background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          Subject {subjectId}
        </Typography>
        <Typography 
          variant="h6"
          sx={{ 
            fontSize: { xs: '0.875rem', md: '1.125rem' },
            fontWeight: 400,
            color: '#475569',
          }}
        >
          Select a training run to start predictions
        </Typography>
      </Box>

      {error && (
        <Box className="mb-6 max-w-2xl">
          <Alert 
            severity="error" 
            action={
              <Button 
                color="inherit" 
                size="small" 
                onClick={loadRuns}
                startIcon={<Refresh />}
              >
                Retry
              </Button>
            }
            className="bg-white/95 shadow-lg"
          >
            {error}
          </Alert>
        </Box>
      )}

      {loading ? (
        <Box className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6">
          {Array.from({ length: 4 }).map((_, index) => (
            <RunCardSkeleton key={index} />
          ))}
        </Box>
      ) : runs.length === 0 ? (
        <Box className="text-center py-12">
          <Typography variant="h6" sx={{ color: '#64748b' }}>
            No runs available for this subject
          </Typography>
        </Box>
          ) : (
            <Box className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6">
              {runs.map((run, index) => (
                <motion.div
                  key={run.run_id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ 
                    duration: 0.4, 
                    delay: index * 0.05,
                    ease: [0.4, 0, 0.2, 1]
                  }}
                >
                  <RunCard run={run} subjectId={subjectId!} />
                </motion.div>
              ))}
            </Box>
          )}
    </Container>
  )
}
