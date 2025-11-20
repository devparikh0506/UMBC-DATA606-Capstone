import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { Container, Box, Typography, Button, IconButton, Card, CardContent, Chip } from '@mui/material'
import { ArrowBack, Refresh } from '@mui/icons-material'
import { motion } from 'framer-motion'
import ResultsSummary from '../components/ResultsSummary'

export default function ResultsPage() {
  const { subjectId, runId } = useParams<{ subjectId: string; runId: string }>()
  const location = useLocation()
  const navigate = useNavigate()
  
  const results = location.state as { accuracy: number; correct: number; total: number } | null

  if (!results) {
    return (
      <Container maxWidth="lg" className="min-h-screen py-12">
        <Card 
          sx={{
            background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)',
            backdropFilter: 'blur(10px)',
          }}
        >
          <CardContent className="p-6 text-center">
            <Typography variant="h6" sx={{ color: '#64748b' }} className="mb-4">
              No results data available
            </Typography>
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
              <Button
                variant="contained"
                onClick={() => navigate(`/subject/${subjectId}/runs`)}
                startIcon={<ArrowBack />}
                sx={{
                  background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #2563eb 0%, #1e40af 100%)',
                  },
                }}
              >
                Back to Runs
              </Button>
            </motion.div>
          </CardContent>
        </Card>
      </Container>
    )
  }

  return (
    <Container maxWidth="lg" className="min-h-screen py-12">
      <Box className="mb-8">
        <Box className="flex items-center gap-4 mb-4">
          <motion.div
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <IconButton
              onClick={() => navigate(`/subject/${subjectId}/runs`)}
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
            className="font-bold"
            sx={{ 
              fontSize: { xs: '1.75rem', md: '2.5rem' },
              background: 'linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #60a5fa 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            Results
          </Typography>
        </Box>
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
        </Box>
        <Box className="flex gap-2">
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <Button
              variant="outlined"
              onClick={() => navigate(`/subject/${subjectId}/run/${runId}/predict`)}
              startIcon={<Refresh />}
              sx={{
                borderColor: '#3b82f6',
                color: '#3b82f6',
                '&:hover': {
                  borderColor: '#2563eb',
                  backgroundColor: 'rgba(59, 130, 246, 0.08)',
                },
              }}
            >
              Run Again
            </Button>
          </motion.div>
          <motion.div
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <Button
              variant="outlined"
              onClick={() => navigate(`/subject/${subjectId}/runs`)}
              startIcon={<ArrowBack />}
              sx={{
                borderColor: '#3b82f6',
                color: '#3b82f6',
                '&:hover': {
                  borderColor: '#2563eb',
                  backgroundColor: 'rgba(59, 130, 246, 0.08)',
                },
              }}
            >
              Back to Runs
            </Button>
          </motion.div>
        </Box>
      </Box>

      <ResultsSummary 
        accuracy={results.accuracy}
        correct={results.correct}
        total={results.total}
      />
    </Container>
  )
}
