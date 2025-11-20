import { Box, Typography } from '@mui/material'
import { motion, AnimatePresence } from 'framer-motion'

interface SmileyAnimationProps {
  prediction: number | null
  correct: boolean | null
}

export default function SmileyAnimation({ prediction, correct }: SmileyAnimationProps) {
  const position = prediction === 0 ? -100 : prediction === 1 ? 100 : 0
  const face = correct === true ? '😊' : correct === false ? '😢' : '😐'

  return (
    <Box className="w-full">
      <Box className="relative h-40 mb-10">
        <motion.div
          className="absolute top-1/2 left-1/2"
          initial={{ x: '-50%', y: '-50%' }}
          animate={{ 
            x: `calc(-50% + ${position}px)`, 
            y: '-50%',
          }}
          transition={{ 
            type: 'spring',
            stiffness: 100,
            damping: 15,
            duration: 0.6
          }}
        >
          <AnimatePresence mode="wait">
            <motion.div
              key={face}
              initial={{ scale: 0.8, opacity: 0, rotate: -10 }}
              animate={{ scale: 1, opacity: 1, rotate: 0 }}
              exit={{ scale: 0.8, opacity: 0, rotate: 10 }}
              transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
              style={{
                filter: 'drop-shadow(0 8px 16px rgba(0, 0, 0, 0.1))',
              }}
            >
              <Typography
                variant="h1"
                sx={{ 
                  fontSize: '6rem',
                  lineHeight: 1,
                  userSelect: 'none',
                }}
              >
                {face}
              </Typography>
            </motion.div>
          </AnimatePresence>
        </motion.div>
      </Box>
      
      <Box 
        className="relative w-full max-w-md mx-auto"
        sx={{
          position: 'relative',
          height: '8px',
          background: 'linear-gradient(90deg, #e0e7ff 0%, #c7d2fe 50%, #e0e7ff 100%)',
          borderRadius: '4px',
          boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.1)',
        }}
      >
        <Box 
          className="absolute left-1/2 top-0 bottom-0 transform -translate-x-1/2"
          sx={{
            width: '2px',
            background: 'linear-gradient(180deg, #64748b 0%, #475569 100%)',
            height: '100%',
          }}
        />
        <Box 
          className="absolute left-0 top-1/2 transform -translate-y-1/2 -translate-x-1/2"
          sx={{
            px: 1.5,
            py: 0.5,
            borderRadius: '8px',
            background: 'rgba(255, 255, 255, 0.9)',
            border: '1px solid rgba(100, 116, 139, 0.2)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          }}
        >
          <Typography 
            variant="body2" 
            sx={{ 
              color: '#475569', 
              fontWeight: 700,
              fontSize: '0.875rem',
              whiteSpace: 'nowrap',
            }}
          >
            Left
          </Typography>
        </Box>
        <Box 
          className="absolute right-0 top-1/2 transform -translate-y-1/2 translate-x-1/2"
          sx={{
            px: 1.5,
            py: 0.5,
            borderRadius: '8px',
            background: 'rgba(255, 255, 255, 0.9)',
            border: '1px solid rgba(100, 116, 139, 0.2)',
            boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          }}
        >
          <Typography 
            variant="body2" 
            sx={{ 
              color: '#475569', 
              fontWeight: 700,
              fontSize: '0.875rem',
              whiteSpace: 'nowrap',
            }}
          >
            Right
          </Typography>
        </Box>
      </Box>
    </Box>
  )
}
