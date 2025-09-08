import { NextResponse } from 'next/server'
import { readFileSync } from 'fs'
import { join } from 'path'

export async function GET() {
  try {
    const filePath = join(process.cwd(), '..', 'results', 'metrics', 'drivers_models_evaluation.csv')
    const fileContent = readFileSync(filePath, 'utf-8')
    
    return new NextResponse(fileContent, {
      headers: {
        'Content-Type': 'text/csv',
      },
    })
  } catch (error) {
    console.error('Error reading drivers model evaluation:', error)
    return NextResponse.json({ error: 'Failed to load drivers model evaluation' }, { status: 500 })
  }
}