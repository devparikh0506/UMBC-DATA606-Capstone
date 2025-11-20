import { useNavigate } from 'react-router-dom'
import { Card, CardContent, Typography, Box } from '@mui/material'
import { Person } from '@mui/icons-material'
import { motion } from 'framer-motion'

interface SubjectCardProps {
  subjectId: string
}

export default function SubjectCard({ subjectId }: SubjectCardProps) {
  const navigate = useNavigate()

  const handleClick = () => {
    navigate(`/subject/${subjectId}/runs`)
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
        <Box className="flex items-center gap-4 mb-4">
          <Box 
            className="w-16 h-16 rounded-xl flex items-center justify-center shadow-md"
            sx={{
              background: 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%)',
            }}
          >
            <Person sx={{ fontSize: 32, color: 'white' }} />
          </Box>
          <Box className="flex-1 min-w-0">
            <Typography 
              variant="h5" 
              className="font-bold text-gray-900 truncate"
              sx={{ fontWeight: 700, fontSize: '1.25rem' }}
            >
              {subjectId}
            </Typography>
            <Typography 
              variant="body2" 
              className="text-gray-500 mt-1"
              sx={{ fontSize: '0.875rem' }}
            >
              Training Runs
            </Typography>
          </Box>
        </Box>
        <Box className="mt-5 pt-4 border-t border-gray-100">
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
            View Details
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
