import { Skeleton, Box, Card, CardContent } from '@mui/material'

interface SkeletonLoaderProps {
  variant?: 'card' | 'text' | 'circular' | 'rectangular'
  width?: number | string
  height?: number | string
  count?: number
}

export function SubjectCardSkeleton() {
  return (
    <Card className="w-full bg-white border-0 shadow-lg" sx={{ borderRadius: '16px' }}>
      <CardContent className="p-6">
        <Box className="flex items-center gap-4 mb-4">
          <Skeleton variant="rectangular" width={64} height={64} sx={{ borderRadius: '12px' }} />
          <Box className="flex-1">
            <Skeleton variant="text" width="60%" height={32} sx={{ mb: 1 }} />
            <Skeleton variant="text" width="40%" height={20} />
          </Box>
        </Box>
        <Skeleton variant="rectangular" width="100%" height={1} className="mb-4" />
        <Skeleton variant="text" width="30%" height={20} />
      </CardContent>
    </Card>
  )
}

export function RunCardSkeleton() {
  return (
    <Card className="w-full bg-white border-0 shadow-lg" sx={{ borderRadius: '16px' }}>
      <CardContent className="p-6">
        <Box className="flex items-center justify-between mb-4">
          <Box className="flex-1">
            <Skeleton variant="text" width="50%" height={32} sx={{ mb: 1 }} />
            <Skeleton variant="text" width="40%" height={20} />
          </Box>
          <Skeleton variant="rectangular" width={56} height={56} sx={{ borderRadius: '12px' }} />
        </Box>
        <Box className="mt-5 pt-4 border-t border-gray-100 flex items-center justify-between">
          <Skeleton variant="rectangular" width={80} height={24} sx={{ borderRadius: '12px' }} />
          <Skeleton variant="text" width="40%" height={20} />
        </Box>
      </CardContent>
    </Card>
  )
}

export function PredictionSkeleton() {
  return (
    <Box className="space-y-4">
      <Skeleton variant="rectangular" width="100%" height={200} className="rounded-lg" />
      <Box className="space-y-2">
        <Skeleton variant="text" width="60%" height={32} />
        <Skeleton variant="text" width="80%" height={24} />
        <Skeleton variant="text" width="70%" height={24} />
      </Box>
    </Box>
  )
}

export default function SkeletonLoader({ 
  variant = 'text', 
  width = '100%', 
  height = 20,
  count = 1 
}: SkeletonLoaderProps) {
  return (
    <>
      {Array.from({ length: count }).map((_, index) => (
        <Skeleton
          key={index}
          variant={variant}
          width={width}
          height={height}
          className="mb-2"
        />
      ))}
    </>
  )
}

