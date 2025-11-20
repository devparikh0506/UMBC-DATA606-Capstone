const API_BASE_URL = '/api'

export interface Subject {
  subjects: string[]
}

export interface Run {
  run_id: string
  session_id: string
  filename: string
}

export interface RunInfo {
  n_trials: number
  n_channels: number
  n_times: number
  sampling_rate: number
  duration_per_trial: number
}

export async function getSubjects(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/subjects/`)
  if (!response.ok) {
    throw new Error('Failed to fetch subjects')
  }
  const data: Subject = await response.json()
  return data.subjects
}

export async function getRuns(subjectId: string): Promise<Run[]> {
  const response = await fetch(`${API_BASE_URL}/subjects/${subjectId}/runs/`)
  if (!response.ok) {
    throw new Error(`Failed to fetch runs for subject ${subjectId}`)
  }
  const data: { runs: Run[] } = await response.json()
  return data.runs
}

export async function getRunInfo(subjectId: string, runId: string): Promise<RunInfo> {
  const response = await fetch(`${API_BASE_URL}/subjects/${subjectId}/runs/${runId}/info/`)
  if (!response.ok) {
    throw new Error(`Failed to fetch run info for ${subjectId}/${runId}`)
  }
  return await response.json()
}

