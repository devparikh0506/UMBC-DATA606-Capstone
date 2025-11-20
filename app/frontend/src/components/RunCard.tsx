import { useNavigate } from 'react-router-dom'
import { Card, CardContent, Typography, Box, Chip } from '@mui/material'
import { PlayArrow, Schedule } from '@mui/icons-material'
import { motion } from 'framer-motion'

interface RunCardProps {
  run: {
    run_id: string
    session_id: string
    filename: string
  }
  subjectId: string
}

export default function RunCard({ run, subjectId }: RunCardProps) {
  const navigate = useNavigate()

  const handleClick = () => {
    navigate(`/subject/${subjectId}/run/${run.run_id}/predict`)
  }

  return (
    <motion.div
      whileHover={{ 
        scale: 1.02, 
        y: -6,
        transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] }
      }}
      whileTap={{ scale: 0.98 }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
    >
      <Card
        onClick={handleClick}
        className="w-full cursor-pointer border-0 shadow-lg"
        sx={{
          borderRadius: '16px',
          background: 'linear-gradient(135deg, #ffffff 0%, #f0f9ff 50%, #e0f2fe 100%)',
          transition: 'box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            boxShadow: '0 20px 40px rgba(59, 130, 246, 0.15)',
          },
        }}
      >
      <CardContent className="p-6">
        <Box className="flex items-center justify-between mb-4">
          <Box className="flex-1 min-w-0">
            <Typography 
              variant="h5" 
              className="font-bold text-gray-900 mb-1"
              sx={{ fontWeight: 700, fontSize: '1.25rem' }}
            >
              Run {run.run_id}
            </Typography>
            <Box className="flex items-center gap-1.5 text-gray-600">
              <Schedule sx={{ fontSize: 16 }} />
              <Typography variant="body2" sx={{ fontSize: '0.875rem' }}>
                Session {run.session_id}
              </Typography>
            </Box>
          </Box>
          <Box 
            className="w-14 h-14 rounded-xl flex items-center justify-center shadow-md flex-shrink-0"
            sx={{
              background: 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%)',
            }}
          >
            <PlayArrow sx={{ color: 'white', fontSize: 28 }} />
          </Box>
        </Box>
        <Box className="mt-5 pt-4 border-t border-gray-100 flex items-center justify-between">
          <Chip
            label="Training"
            size="small"
            sx={{
              backgroundColor: '#dbeafe',
              color: '#1e40af',
              fontWeight: 600,
              fontSize: '0.75rem',
              height: '24px',
            }}
          />
          <Typography 
            variant="body2" 
            className="text-blue-600 font-semibold flex items-center gap-1"
            sx={{ 
              fontSize: '0.875rem',
              fontWeight: 600,
              '&:hover': {
                color: '#0369a1',
              }
            }}
          >
            Start Prediction
            <Box component="span" sx={{ display: 'inline-block', transition: 'transform 0.2s', '&:hover': { transform: 'translateX(4px)' } }}>
              →
            </Box>
          </Typography>
        </Box>
      </CardContent>
    </Card>
    </motion.div>
  )
}
