import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Container, Typography, Box, Alert, Button, CircularProgress } from '@mui/material'
import { Refresh } from '@mui/icons-material'
import { motion } from 'framer-motion'
import SubjectCard from '../components/SubjectCard'
import { SubjectCardSkeleton } from '../components/SkeletonLoader'
import { getSubjects } from '../services/api'

export default function SubjectSelectionPage() {
  const [subjects, setSubjects] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    loadSubjects()
  }, [])

  const loadSubjects = async () => {
    try {
      setLoading(true)
      const data = await getSubjects()
      setSubjects(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load subjects')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="xl" className="min-h-screen py-10 px-4">
      <Box className="text-center mb-12">
        <Typography 
          variant="h2" 
          className="font-bold mb-4"
          sx={{ 
            fontSize: { xs: '2rem', sm: '2.5rem', md: '3.5rem' },
            fontWeight: 700,
            letterSpacing: '-0.02em',
            background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          BCI Real-time Prediction
        </Typography>
        <Typography 
          variant="h6" 
          className="mb-8"
          sx={{ 
            fontSize: { xs: '1rem', md: '1.25rem' },
            fontWeight: 400,
            color: '#475569',
          }}
        >
          Select a subject to view training runs
        </Typography>
      </Box>

      {error && (
        <Box className="mb-6 max-w-2xl mx-auto">
          <Alert 
            severity="error" 
            action={
              <Button 
                color="inherit" 
                size="small" 
                onClick={loadSubjects}
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
          {Array.from({ length: 8 }).map((_, index) => (
            <SubjectCardSkeleton key={index} />
          ))}
        </Box>
      ) : subjects.length === 0 ? (
        <Box className="text-center py-12">
          <Typography variant="h6" sx={{ color: '#64748b' }}>
            No subjects available
          </Typography>
        </Box>
      ) : (
        <Box className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6">
          {subjects.map((subjectId, index) => (
            <motion.div
              key={subjectId}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ 
                duration: 0.4, 
                delay: index * 0.05,
                ease: [0.4, 0, 0.2, 1]
              }}
            >
              <SubjectCard subjectId={subjectId} />
            </motion.div>
          ))}
        </Box>
      )}
    </Container>
  )
}
