import { NextResponse } from 'next/server'
import { readFileSync } from 'fs'
import { join } from 'path'

export async function GET() {
  try {
    const filePath = join(process.cwd(), '..', 'data', 'grab_merchants_dataset_refined_score.csv')
    const fileContent = readFileSync(filePath, 'utf-8')
    
    // Return raw CSV content for Papa Parse to handle
    return new NextResponse(fileContent, {
      headers: {
        'Content-Type': 'text/csv',
        'Cache-Control': 'public, max-age=3600'
      }
    })
  } catch (error) {
    console.error('Error reading merchants CSV:', error)
    return NextResponse.json({ 
      success: false,
      error: 'Failed to load merchants data' 
    }, { status: 500 })
  }
}
