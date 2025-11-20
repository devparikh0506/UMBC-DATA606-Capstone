import { Card, CardContent, Box, Typography, Chip } from '@mui/material'
import { CheckCircle, Cancel, TrendingUp } from '@mui/icons-material'
import { motion, AnimatePresence } from 'framer-motion'

interface PredictionIndicatorProps {
  prediction: number | null
  confidence: number | null
  groundTruth: number | null
  correct: boolean | null
}

export default function PredictionIndicator({
  prediction,
  confidence,
  groundTruth,
  correct
}: PredictionIndicatorProps) {
  if (prediction === null) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
      >
        <Card 
          sx={{
            background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
            border: '2px dashed #cbd5e1',
            borderRadius: '16px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
          }}
        >
          <CardContent className="p-8 text-center">
            <Typography 
              variant="body1" 
              sx={{ 
                color: '#64748b',
                fontSize: '1.1rem',
                fontWeight: 500,
              }}
            >
              Waiting for prediction...
            </Typography>
            <Typography 
              variant="caption" 
              sx={{ 
                color: '#94a3b8',
                fontSize: '0.875rem',
                mt: 1,
                display: 'block',
              }}
            >
              Start predictions to see results
            </Typography>
          </CardContent>
        </Card>
      </motion.div>
    )
  }

  const predictionLabel = prediction === 0 ? 'Left' : 'Right'
  const groundTruthLabel = groundTruth !== null ? (groundTruth === 0 ? 'Left' : 'Right') : 'Unknown'
  const confidencePercent = confidence ? (confidence * 100).toFixed(1) : 'N/A'

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 20 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
    >
      <Card 
        sx={{
          background: correct === true 
            ? 'linear-gradient(135deg, #ffffff 0%, #f0fdf4 50%, #dcfce7 100%)'
            : correct === false 
            ? 'linear-gradient(135deg, #ffffff 0%, #fef2f2 50%, #fee2e2 100%)'
            : 'linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #f1f5f9 100%)',
          border: `2px solid ${
            correct === true 
              ? '#86efac' 
              : correct === false 
              ? '#fca5a5' 
              : '#e2e8f0'
          }`,
          borderRadius: '16px',
          boxShadow: correct === true
            ? '0 8px 24px rgba(34, 197, 94, 0.15)'
            : correct === false
            ? '0 8px 24px rgba(239, 68, 68, 0.15)'
            : '0 4px 12px rgba(0, 0, 0, 0.08)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
      >
        <CardContent className="p-6">
        <Box className="space-y-5">
          <Box 
            className="flex items-center justify-between"
            sx={{
              p: 2,
              borderRadius: '12px',
              background: 'rgba(255, 255, 255, 0.6)',
            }}
          >
            <Typography 
              variant="body1" 
              sx={{ 
                color: '#475569', 
                fontWeight: 600,
                fontSize: '0.95rem',
              }}
            >
              Prediction
            </Typography>
            <Chip
              label={predictionLabel}
              sx={{
                fontWeight: 700,
                fontSize: '0.875rem',
                px: 1,
                height: 32,
                background: prediction === 0 
                  ? 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)'
                  : 'linear-gradient(135deg, #fed7aa 0%, #fdba74 100%)',
                color: prediction === 0 ? '#1e40af' : '#9a3412',
                border: `1px solid ${prediction === 0 ? '#93c5fd' : '#fb923c'}`,
                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
              }}
            />
          </Box>

          <Box 
            className="flex items-center justify-between"
            sx={{
              p: 2,
              borderRadius: '12px',
              background: 'rgba(255, 255, 255, 0.6)',
            }}
          >
            <Typography 
              variant="body1" 
              sx={{ 
                color: '#475569', 
                fontWeight: 600,
                fontSize: '0.95rem',
              }}
            >
              Confidence
            </Typography>
            <Box className="flex items-center gap-2">
              <TrendingUp 
                sx={{ 
                  color: '#3b82f6', 
                  fontSize: 20,
                }} 
              />
              <Typography 
                variant="h6" 
                sx={{ 
                  fontWeight: 700, 
                  color: '#1e293b',
                  fontSize: '1.25rem',
                }}
              >
                {confidencePercent}%
              </Typography>
            </Box>
          </Box>

          <Box 
            className="flex items-center justify-between"
            sx={{
              p: 2,
              borderRadius: '12px',
              background: 'rgba(255, 255, 255, 0.6)',
            }}
          >
            <Typography 
              variant="body1" 
              sx={{ 
                color: '#475569', 
                fontWeight: 600,
                fontSize: '0.95rem',
              }}
            >
              Ground Truth
            </Typography>
            <Typography 
              variant="body1" 
              sx={{ 
                fontWeight: 600, 
                color: '#1e293b',
                fontSize: '1rem',
              }}
            >
              {groundTruthLabel}
            </Typography>
          </Box>

          <Box 
            className="pt-4 border-t"
            sx={{
              borderColor: correct === true 
                ? 'rgba(34, 197, 94, 0.2)' 
                : correct === false 
                ? 'rgba(239, 68, 68, 0.2)' 
                : 'rgba(226, 232, 240, 1)',
            }}
          >
            <Box 
              className="flex items-center justify-between"
              sx={{
                p: 2,
                borderRadius: '12px',
                background: correct === true
                  ? 'rgba(220, 252, 231, 0.5)'
                  : correct === false
                  ? 'rgba(254, 226, 226, 0.5)'
                  : 'rgba(248, 250, 252, 0.6)',
              }}
            >
              <Typography 
                variant="body1" 
                sx={{ 
                  color: '#475569', 
                  fontWeight: 600,
                  fontSize: '0.95rem',
                }}
              >
                Status
              </Typography>
              <Chip
                icon={correct ? <CheckCircle /> : <Cancel />}
                label={correct ? 'Correct' : 'Incorrect'}
                sx={{
                  fontWeight: 600,
                  fontSize: '0.875rem',
                  px: 1,
                  height: 32,
                  background: correct
                    ? 'linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)'
                    : 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
                  color: correct ? '#166534' : '#991b1b',
                  border: `1px solid ${correct ? '#86efac' : '#fca5a5'}`,
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                  '& .MuiChip-icon': {
                    color: correct ? '#16a34a' : '#dc2626',
                  },
                }}
              />
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
    </motion.div>
  )
}
