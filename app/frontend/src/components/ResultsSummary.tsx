import { Box, Typography, Card, CardContent, Grid } from '@mui/material'
import { CheckCircle, Assessment, TrendingUp } from '@mui/icons-material'
import { motion } from 'framer-motion'

interface ResultsSummaryProps {
  accuracy: number
  correct: number
  total: number
}

export default function ResultsSummary({ accuracy, correct, total }: ResultsSummaryProps) {
  const accuracyPercent = (accuracy * 100).toFixed(1)
  const incorrect = total - correct

  return (
    <Box className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <Card 
          sx={{
            background: 'linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%)',
            color: 'white',
          }}
        >
          <CardContent className="p-8 text-center">
            <Typography variant="h4" className="font-bold mb-2" sx={{ color: 'white' }}>
              Final Results
            </Typography>
            <Typography variant="h2" className="font-bold mb-4" sx={{ color: 'white' }}>
              {accuracyPercent}%
            </Typography>
            <Typography variant="body1" sx={{ color: 'rgba(255, 255, 255, 0.9)' }}>
              Overall Accuracy
            </Typography>
          </CardContent>
        </Card>
      </motion.div>

      <Grid container spacing={3}>
        <Grid item xs={12} sm={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1, ease: [0.4, 0, 0.2, 1] }}
            whileHover={{ 
              scale: 1.03, 
              y: -4,
              transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] }
            }}
          >
            <Card 
              sx={{
                background: 'linear-gradient(135deg, #ffffff 0%, #f0fdf4 50%, #dcfce7 100%)',
                backdropFilter: 'blur(10px)',
              }}
              className="h-full"
            >
              <CardContent className="p-6 text-center">
                <Box className="flex justify-center mb-4">
                  <Box 
                    className="w-16 h-16 rounded-full flex items-center justify-center"
                    sx={{
                      background: 'linear-gradient(135deg, #86efac 0%, #4ade80 50%, #22c55e 100%)',
                    }}
                  >
                    <CheckCircle className="text-white text-3xl" />
                  </Box>
                </Box>
                <Typography variant="h3" className="font-bold text-gray-800 mb-2">
                  {correct}
                </Typography>
                <Typography variant="body1" className="text-gray-600">
                  Correct Predictions
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2, ease: [0.4, 0, 0.2, 1] }}
            whileHover={{ 
              scale: 1.03, 
              y: -4,
              transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] }
            }}
          >
            <Card 
              sx={{
                background: 'linear-gradient(135deg, #ffffff 0%, #fef2f2 50%, #fee2e2 100%)',
                backdropFilter: 'blur(10px)',
              }}
              className="h-full"
            >
              <CardContent className="p-6 text-center">
                <Box className="flex justify-center mb-4">
                  <Box 
                    className="w-16 h-16 rounded-full flex items-center justify-center"
                    sx={{
                      background: 'linear-gradient(135deg, #fca5a5 0%, #f87171 50%, #ef4444 100%)',
                    }}
                  >
                    <Assessment className="text-white text-3xl" />
                  </Box>
                </Box>
                <Typography variant="h3" className="font-bold text-gray-800 mb-2">
                  {incorrect}
                </Typography>
                <Typography variant="body1" className="text-gray-600">
                  Incorrect Predictions
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={4}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
            whileHover={{ 
              scale: 1.03, 
              y: -4,
              transition: { duration: 0.3, ease: [0.4, 0, 0.2, 1] }
            }}
          >
            <Card 
              sx={{
                background: 'linear-gradient(135deg, #ffffff 0%, #eff6ff 50%, #dbeafe 100%)',
                backdropFilter: 'blur(10px)',
              }}
              className="h-full"
            >
              <CardContent className="p-6 text-center">
                <Box className="flex justify-center mb-4">
                  <Box 
                    className="w-16 h-16 rounded-full flex items-center justify-center"
                    sx={{
                      background: 'linear-gradient(135deg, #93c5fd 0%, #60a5fa 50%, #3b82f6 100%)',
                    }}
                  >
                    <TrendingUp className="text-white text-3xl" />
                  </Box>
                </Box>
                <Typography variant="h3" className="font-bold text-gray-800 mb-2">
                  {total}
                </Typography>
                <Typography variant="body1" className="text-gray-600">
                  Total Predictions
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  )
}
